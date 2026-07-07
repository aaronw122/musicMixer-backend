"""SMS notification service using Twilio."""

import logging
import threading
import time

from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from musicmixer.config import settings

logger = logging.getLogger(__name__)

# Process-wide rolling send budget — a bill-blast circuit breaker across all
# sessions. Backstops the per-session and per-IP limits enforced upstream.
_SMS_BUDGET_PER_HOUR = 100
_SMS_BUDGET_WINDOW_S = 3600.0
_sms_budget_lock = threading.Lock()
_sms_budget_hits: list[float] = []


def _budget_allows() -> bool:
    """Reserve a slot in the rolling hourly send budget; False if exhausted."""
    now = time.monotonic()
    cutoff = now - _SMS_BUDGET_WINDOW_S
    with _sms_budget_lock:
        while _sms_budget_hits and _sms_budget_hits[0] < cutoff:
            _sms_budget_hits.pop(0)
        if len(_sms_budget_hits) >= _SMS_BUDGET_PER_HOUR:
            return False
        _sms_budget_hits.append(now)
        return True


def _mask_phone(phone: object) -> str:
    """Mask a phone number for logging, keeping only leading digit + last 4.

    Never raises: unparseable or too-short input yields a fully-masked
    placeholder so full numbers never reach log storage.
    """
    if not isinstance(phone, str):
        return "***"
    digits = "".join(c for c in phone if c.isdigit())
    if len(digits) < 6:
        return "***"
    prefix = "+" if phone.strip().startswith("+") else ""
    masked_len = len(digits) - 1 - 4
    return f"{prefix}{digits[0]}{'*' * masked_len}{digits[-4:]}"


def _get_client() -> Client:
    """Create a Twilio REST client from settings."""
    return Client(settings.twilio_account_sid, settings.twilio_auth_token)


def send_remix_ready(phone: str, session_id: str) -> bool:
    """Send an SMS notifying the user their remix is ready.

    Args:
        phone: E.164 formatted phone number.
        session_id: The remix session ID for link construction.

    Returns:
        True if SMS was sent successfully, False otherwise.
    """
    if not settings.sms_enabled:
        logger.info("SMS disabled, skipping remix-ready notification to %s", _mask_phone(phone))
        return False

    if not _budget_allows():
        logger.warning("SMS budget exceeded, skipping remix-ready notification to %s", _mask_phone(phone))
        return False

    link = f"{settings.app_base_url}/?listen={session_id}"
    body = f"musicMixer: Your remix is ready! Listen here: {link}"

    try:
        client = _get_client()
        client.messages.create(
            body=body,
            from_=settings.twilio_from_number,
            to=phone,
        )
        logger.info("Sent remix-ready SMS to %s for session %s", _mask_phone(phone), session_id)
        return True
    except TwilioRestException:
        logger.exception("Failed to send remix-ready SMS to %s", _mask_phone(phone))
        return False
    except Exception:
        logger.exception("Unexpected error sending remix-ready SMS to %s", _mask_phone(phone))
        return False


def send_confirmation(phone: str) -> bool:
    """Send a confirmation SMS that we received the phone number.

    Args:
        phone: E.164 formatted phone number.

    Returns:
        True if SMS was sent successfully, False otherwise.
    """
    if not settings.sms_enabled:
        logger.info("SMS disabled, skipping confirmation to %s", _mask_phone(phone))
        return False

    if not _budget_allows():
        logger.warning("SMS budget exceeded, skipping confirmation to %s", _mask_phone(phone))
        return False

    body = "musicMixer: Got it! We'll text you when your remix is ready."

    try:
        client = _get_client()
        client.messages.create(
            body=body,
            from_=settings.twilio_from_number,
            to=phone,
        )
        logger.info("Sent confirmation SMS to %s", _mask_phone(phone))
        return True
    except TwilioRestException:
        logger.exception("Failed to send confirmation SMS to %s", _mask_phone(phone))
        return False
    except Exception:
        logger.exception("Unexpected error sending confirmation SMS to %s", _mask_phone(phone))
        return False
