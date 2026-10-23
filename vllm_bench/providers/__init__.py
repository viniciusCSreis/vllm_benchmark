"""Provider selection; the client runner does not know about Compose internals."""

import json

from .compose import ComposeProvider
from .compose import stop_saved as stop_compose


def get_provider(config, root):
    if config.provider == "local":
        return ComposeProvider(config, root)
    if config.provider == "modal":
        from .modal import ModalProvider

        return ModalProvider(config, root)
    raise ValueError(f"Provider is not implemented: {config.provider}")


def stop_saved(state_path):
    """Recover the saved provider without loading today's experiment config."""
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    if state.get("provider", "local") == "modal":
        from .modal import stop_saved as stop_modal

        stop_modal(state_path)
    else:
        stop_compose(state_path)
