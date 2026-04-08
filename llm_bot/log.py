import logging


logger = logging.getLogger("llm_bot")


def configure_logging(debug: bool = False) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
