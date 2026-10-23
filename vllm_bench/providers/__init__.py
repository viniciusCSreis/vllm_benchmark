"""Provider selection; the client runner does not know about Compose internals."""

from .compose import ComposeProvider


def get_provider(config, root):
    if config.provider == "local":
        return ComposeProvider(config, root)
    raise ValueError(f"Provider is not implemented: {config.provider}")
