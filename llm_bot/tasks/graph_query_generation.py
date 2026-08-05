import json
import re
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.log import logger
from llm_bot.schemas import (
    GraphNodeLabel,
    GraphQueryGenerationRequest,
    GraphQueryGenerationResponse,
    GraphRelationshipType,
)
from llm_bot.tasks.llm_utils import (
    InvalidLLMOutputError,
    create_and_parse_response,
    get_output_text,
    loads_json_output,
)


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "graph_query_generation.txt"
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
NODE_PATTERN = re.compile(rf"\(\s*(?P<variable>{IDENTIFIER})\s*:\s*(?P<label>{IDENTIFIER})(?P<body>[^)]*)\)")
RELATIONSHIP_PATTERN = re.compile(rf"\[\s*(?:(?P<variable>{IDENTIFIER})\s*)?:\s*(?P<type>{IDENTIFIER})(?P<body>[^\]]*)\]")
PROPERTY_ACCESS_PATTERN = re.compile(rf"\b(?P<variable>{IDENTIFIER})\s*\.\s*(?P<property>{IDENTIFIER})\b")
PARAMETER_PATTERN = re.compile(rf"\$(?P<name>{IDENTIFIER})\b")
PROHIBITED_KEYWORDS = {
    "ALTER",
    "CALL",
    "COMMIT",
    "CONSTRAINT",
    "CREATE",
    "CSV",
    "DATABASE",
    "DELETE",
    "DENY",
    "DETACH",
    "DROP",
    "FOREACH",
    "FROM",
    "GRANT",
    "INDEX",
    "LOAD",
    "MERGE",
    "REMOVE",
    "RENAME",
    "REVOKE",
    "ROLLBACK",
    "SELECT",
    "SET",
    "SHOW",
    "START",
    "STOP",
    "TERMINATE",
    "TRANSACTION",
    "UNION",
    "USE",
    "YIELD",
}
PROHIBITED_DYNAMIC_FUNCTIONS = {"cypher", "keys", "labels", "properties", "type"}
LITERAL_KEYWORDS = {"FALSE", "NULL", "TRUE"}


def load_graph_query_generation_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_graph_query_generation_messages(
    request: GraphQueryGenerationRequest,
) -> list[dict[str, str]]:
    user_payload = {
        "question": request.question,
        "graph_name": request.graph_name,
        "schema": request.schema.model_dump(exclude_none=True),
    }
    return [
        {"role": "system", "content": load_graph_query_generation_prompt()},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
    ]


def get_graph_query_generation_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "graph_query_generation_response",
        # Parameter names are generated from the question, so the object cannot
        # enumerate fixed properties. Local parsing and validation remain strict.
        "strict": False,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["cypher", "parameters", "explanation"],
            "properties": {
                "cypher": {"type": "string", "minLength": 1},
                "parameters": {"type": "object", "additionalProperties": True},
                "explanation": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                },
            },
        },
    }


def _raise_invalid(message: str) -> None:
    raise InvalidLLMOutputError(message)


def _validate_query_shape(cypher: str, maximum_limit: int) -> None:
    if any(marker in cypher for marker in (";", "//", "/*", "*/", "'", '"', "`")):
        _raise_invalid("Cypher must be one statement without comments, quoted literals, or quoted identifiers")

    keywords = {token.upper() for token in re.findall(IDENTIFIER, cypher)}
    prohibited = sorted(keywords & PROHIBITED_KEYWORDS)
    if prohibited:
        _raise_invalid(f"Cypher contains prohibited clauses or keywords: {', '.join(prohibited)}")
    literal_keywords = sorted(keywords & LITERAL_KEYWORDS)
    if literal_keywords:
        _raise_invalid("Cypher must use parameters instead of literal values: " + ", ".join(literal_keywords))

    dynamic_functions = sorted(
        function for function in PROHIBITED_DYNAMIC_FUNCTIONS if re.search(rf"\b{function}\s*\(", cypher, flags=re.IGNORECASE)
    )
    if dynamic_functions:
        _raise_invalid("Cypher contains prohibited dynamic schema access: " + ", ".join(dynamic_functions))
    if re.search(r"\b(?:COLLECT|COUNT|EXISTS)\s*\{", cypher, flags=re.IGNORECASE):
        _raise_invalid("Cypher must not contain subqueries")

    if not re.search(r"\bMATCH\b", cypher, flags=re.IGNORECASE):
        _raise_invalid("Cypher must contain MATCH")
    if not re.search(r"\bRETURN\b", cypher, flags=re.IGNORECASE):
        _raise_invalid("Cypher must contain RETURN")

    limit_matches = list(re.finditer(r"\bLIMIT\s+(\d+)\b", cypher, flags=re.IGNORECASE))
    if len(limit_matches) != 1:
        _raise_invalid("Cypher must contain exactly one integer LIMIT")
    limit = int(limit_matches[0].group(1))
    if limit < 1 or limit > maximum_limit:
        _raise_invalid(f"Cypher LIMIT must be between 1 and {maximum_limit}")
    if cypher[limit_matches[0].end() :].strip():
        _raise_invalid("Cypher LIMIT must be the final clause")

    cypher_without_limit = cypher[: limit_matches[0].start(1)] + cypher[limit_matches[0].end(1) :]
    if re.search(r"(?<![$A-Za-z_])\d+(?:\.\d+)?\b", cypher_without_limit):
        _raise_invalid("Cypher must use parameters instead of numeric literal values")


def _property_names(owner: GraphNodeLabel | GraphRelationshipType) -> set[str]:
    return {prop.name for prop in owner.properties}


def _validate_inline_properties(
    body: str,
    *,
    allowed_properties: set[str],
    owner_description: str,
) -> None:
    for property_map in re.findall(r"\{([^{}]*)\}", body):
        property_names = re.findall(rf"\b({IDENTIFIER})\s*:", property_map)
        unknown = sorted(set(property_names) - allowed_properties)
        if unknown:
            _raise_invalid(f"Cypher uses unknown properties on {owner_description}: {', '.join(unknown)}")


def _validate_schema_references(
    cypher: str,
    request: GraphQueryGenerationRequest,
) -> None:
    nodes_by_label = {node.label: node for node in request.schema.node_labels}
    relationships_by_type = {relationship.type: relationship for relationship in request.schema.relationship_types}
    variables: dict[str, GraphNodeLabel | GraphRelationshipType] = {}

    node_matches = list(NODE_PATTERN.finditer(cypher))
    if not node_matches:
        _raise_invalid("Cypher MATCH patterns must use explicitly labeled nodes")

    match_clauses = list(
        re.finditer(
            r"\bMATCH\b(?P<body>.*?)(?=\b(?:WHERE|WITH|RETURN|OPTIONAL\s+MATCH|"
            r"ORDER\s+BY|SKIP|LIMIT)\b|$)",
            cypher,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    node_starts = {match.start() for match in node_matches}
    for clause in match_clauses:
        if any(cypher[index] == "(" and index not in node_starts for index in range(clause.start("body"), clause.end("body"))):
            _raise_invalid("Every node in a MATCH pattern must use one explicit allowed label")

    for match in node_matches:
        variable = match.group("variable")
        label = match.group("label")
        node_schema = nodes_by_label.get(label)
        if node_schema is None:
            _raise_invalid(f"Cypher uses unknown node label: {label}")
        node_body_without_maps = re.sub(r"\{[^{}]*\}", "", match.group("body"))
        if ":" in node_body_without_maps:
            _raise_invalid("Cypher nodes must use exactly one explicit allowed label")
        existing_owner = variables.get(variable)
        if existing_owner is not None and existing_owner is not node_schema:
            _raise_invalid(f"Cypher variable {variable} is assigned incompatible schema types")
        variables[variable] = node_schema
        _validate_inline_properties(
            match.group("body"),
            allowed_properties=_property_names(node_schema),
            owner_description=f"node label {label}",
        )

    relationship_matches = list(RELATIONSHIP_PATTERN.finditer(cypher))
    if cypher.count("[") != len(relationship_matches) or cypher.count("]") != len(relationship_matches):
        _raise_invalid("Cypher relationships must use one explicit allowed relationship type")

    for match in relationship_matches:
        relationship_type = match.group("type")
        relationship_schema = relationships_by_type.get(relationship_type)
        if relationship_schema is None:
            _raise_invalid(f"Cypher uses unknown relationship type: {relationship_type}")
        relationship_body_without_maps = re.sub(r"\{[^{}]*\}", "", match.group("body"))
        if any(marker in relationship_body_without_maps for marker in (":", "|", "$")):
            _raise_invalid(f"Relationship {relationship_type} must use one static allowed type")
        if variable := match.group("variable"):
            existing_owner = variables.get(variable)
            if existing_owner is not None and existing_owner is not relationship_schema:
                _raise_invalid(f"Cypher variable {variable} is assigned incompatible schema types")
            variables[variable] = relationship_schema
        _validate_inline_properties(
            match.group("body"),
            allowed_properties=_property_names(relationship_schema),
            owner_description=f"relationship type {relationship_type}",
        )

        left_nodes = [node for node in node_matches if node.end() <= match.start()]
        right_nodes = [node for node in node_matches if node.start() >= match.end()]
        if not left_nodes or not right_nodes:
            _raise_invalid(f"Relationship {relationship_type} must connect two labeled nodes")
        left_node = left_nodes[-1]
        right_node = right_nodes[0]
        left_connector = cypher[left_node.end() : match.start()].strip()
        right_connector = cypher[match.end() : right_node.start()].strip()
        if left_connector == "-" and right_connector == "->":
            source_label = left_node.group("label")
            target_label = right_node.group("label")
        elif left_connector == "<-" and right_connector == "-":
            source_label = right_node.group("label")
            target_label = left_node.group("label")
        else:
            _raise_invalid(f"Relationship {relationship_type} must have one directed arrow")
        if source_label not in relationship_schema.source_labels or target_label not in relationship_schema.target_labels:
            _raise_invalid(f"Relationship {relationship_type} does not allow source label {source_label} and target label {target_label}")

    for match in PROPERTY_ACCESS_PATTERN.finditer(cypher):
        variable = match.group("variable")
        property_name = match.group("property")
        owner = variables.get(variable)
        if owner is None:
            _raise_invalid(f"Cypher accesses property on unknown variable: {variable}")
        if property_name not in _property_names(owner):
            _raise_invalid(f"Cypher uses unknown property {property_name} on variable {variable}")

    return_match = re.search(
        r"\bRETURN\b(?P<body>.*?)(?=\b(?:ORDER\s+BY|SKIP|LIMIT)\b)",
        cypher,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if return_match is None:
        _raise_invalid("Cypher must have one result-producing RETURN clause")
    return_body = return_match.group("body")
    return_body_without_count_all = re.sub(
        r"\bcount\s*\(\s*\*\s*\)",
        "",
        return_body,
        flags=re.IGNORECASE,
    )
    if "*" in return_body_without_count_all:
        _raise_invalid("Cypher must not return all values from the query scope")
    for variable in variables:
        return_body_without_count = re.sub(
            rf"\bcount\s*\(\s*(?:DISTINCT\s+)?{re.escape(variable)}\s*\)",
            "",
            return_body,
            flags=re.IGNORECASE,
        )
        if re.search(
            rf"(?<![.\w]){re.escape(variable)}\b(?!\s*(?:\.|:))",
            return_body_without_count,
        ):
            _raise_invalid(f"Cypher must not return whole graph variable: {variable}")
        if re.search(
            rf"(?<![.\w]){re.escape(variable)}\s+AS\s+{IDENTIFIER}\b",
            cypher,
            flags=re.IGNORECASE,
        ):
            _raise_invalid(f"Cypher must not alias whole graph variable: {variable}")
    if not re.search(r"\bAS\s+result\s*$", return_body.strip(), flags=re.IGNORECASE):
        _raise_invalid("Cypher must return exactly one value aliased as result")


def _validate_parameters(cypher: str, parameters: dict[str, Any]) -> None:
    invalid_parameter_names = sorted(name for name in parameters if not re.fullmatch(IDENTIFIER, name))
    if invalid_parameter_names:
        _raise_invalid("Parameters contain invalid names: " + ", ".join(invalid_parameter_names))

    placeholders = set(PARAMETER_PATTERN.findall(cypher))
    parameter_names = set(parameters)
    missing_parameters = sorted(placeholders - parameter_names)
    extra_parameters = sorted(parameter_names - placeholders)
    if missing_parameters:
        _raise_invalid("Cypher placeholders have no parameter values: " + ", ".join(missing_parameters))
    if extra_parameters:
        _raise_invalid("Parameters are not used by Cypher: " + ", ".join(extra_parameters))


def validate_graph_query_generation(
    response: GraphQueryGenerationResponse,
    request: GraphQueryGenerationRequest,
) -> GraphQueryGenerationResponse:
    cypher = response.cypher.strip()
    _validate_query_shape(cypher, request.schema.maximum_limit)
    _validate_schema_references(cypher, request)
    _validate_parameters(cypher, response.parameters)
    response.cypher = cypher
    return response


def parse_graph_query_generation_response(
    response_data: dict[str, Any],
    request: GraphQueryGenerationRequest,
) -> GraphQueryGenerationResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw graph query generation output: %s", output_text)
    parsed_output = loads_json_output(output_text)
    response = GraphQueryGenerationResponse.model_validate(parsed_output)
    return validate_graph_query_generation(response, request)


async def generate_graph_query(
    request: GraphQueryGenerationRequest,
    client: LLMClient | None = None,
) -> GraphQueryGenerationResponse:
    llm_client = client or LLMClient(
        reasoning_effort=request.reasoning_effort,
        thinking_budget_tokens=request.thinking_budget_tokens,
    )
    system_message, user_message = build_graph_query_generation_messages(request)
    return await create_and_parse_response(
        client=llm_client,
        task_name="graph query generation",
        user_input=user_message["content"],
        system_input=system_message["content"],
        response_format=get_graph_query_generation_response_format(),
        parse_response=lambda response_data: parse_graph_query_generation_response(response_data, request),
    )
