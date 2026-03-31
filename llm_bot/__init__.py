__all__ = ["__version__"]

try:
    from importlib.metadata import version

    __version__ = version("llm-bot")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
