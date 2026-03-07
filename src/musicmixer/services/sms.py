"""SMS notification service using Twilio."""

import logging

from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from musicmixer.config import settings

logger = logging.getLogger(__name__)


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
        logger.info("SMS disabled, skipping remix-ready notification to %s", phone)
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
        logger.info("Sent remix-ready SMS to %s for session %s", phone, session_id)
        return True
    except TwilioRestException:
        logger.exception("Failed to send remix-ready SMS to %s", phone)
        return False
    except Exception:
        logger.exception("Unexpected error sending remix-ready SMS to %s", phone)
        return False


def send_confirmation(phone: str) -> bool:
    """Send a confirmation SMS that we received the phone number.

    Args:
        phone: E.164 formatted phone number.

    Returns:
        True if SMS was sent successfully, False otherwise.
    """
    if not settings.sms_enabled:
        logger.info("SMS disabled, skipping confirmation to %s", phone)
        return False

    body = "musicMixer: Got it! We'll text you when your remix is ready."

    try:
        client = _get_client()
        client.messages.create(
            body=body,
            from_=settings.twilio_from_number,
            to=phone,
        )
        logger.info("Sent confirmation SMS to %s", phone)
        return True
    except TwilioRestException:
        logger.exception("Failed to send confirmation SMS to %s", phone)
        return False
    except Exception:
        logger.exception("Unexpected error sending confirmation SMS to %s", phone)
        return False
