"""SMS notification service (stub).

Full implementation is in the parallel branch (Phase 1B).
This stub exists so that imports work for the notify endpoint and pipeline hook.
"""


def send_remix_ready(phone: str, session_id: str) -> bool:
    """Send 'remix ready' SMS. Returns True on success."""
    raise NotImplementedError("SMS service not yet configured")


def send_confirmation(phone: str) -> bool:
    """Send confirmation SMS. Returns True on success."""
    raise NotImplementedError("SMS service not yet configured")
