"""Tests for the SMS notification service."""

import logging
from unittest.mock import patch, MagicMock

import pytest

from twilio.base.exceptions import TwilioRestException


@pytest.fixture
def sms_settings():
    """Patch settings for SMS tests."""
    with patch("musicmixer.services.sms.settings") as mock_settings:
        mock_settings.sms_enabled = True
        mock_settings.twilio_account_sid = "ACtest123"
        mock_settings.twilio_auth_token = "test_token"
        mock_settings.twilio_from_number = "+15550001111"
        mock_settings.app_base_url = "https://mixer.awill.co"
        yield mock_settings


@pytest.fixture
def mock_twilio_client():
    """Mock the Twilio Client."""
    with patch("musicmixer.services.sms.Client") as MockClient:
        client_instance = MagicMock()
        MockClient.return_value = client_instance
        yield client_instance


class TestSendRemixReady:
    def test_sends_correct_message(self, sms_settings, mock_twilio_client):
        """Should send SMS with correct body and recipient."""
        from musicmixer.services.sms import send_remix_ready

        result = send_remix_ready("+15551234567", "abc-123")

        assert result is True
        mock_twilio_client.messages.create.assert_called_once_with(
            body="musicMixer: Your remix is ready! Listen here: https://mixer.awill.co/?listen=abc-123",
            from_="+15550001111",
            to="+15551234567",
        )

    def test_returns_false_when_disabled(self, sms_settings, mock_twilio_client):
        """Should not send SMS when sms_enabled is False."""
        from musicmixer.services.sms import send_remix_ready

        sms_settings.sms_enabled = False
        result = send_remix_ready("+15551234567", "abc-123")

        assert result is False
        mock_twilio_client.messages.create.assert_not_called()

    def test_catches_twilio_exception(self, sms_settings, mock_twilio_client, caplog):
        """Should catch TwilioRestException and return False."""
        from musicmixer.services.sms import send_remix_ready

        mock_twilio_client.messages.create.side_effect = TwilioRestException(
            status=400, uri="/Messages", msg="Invalid phone"
        )

        with caplog.at_level(logging.ERROR):
            result = send_remix_ready("+15551234567", "abc-123")

        assert result is False
        assert "Failed to send remix-ready SMS" in caplog.text

    def test_catches_generic_exception(self, sms_settings, mock_twilio_client, caplog):
        """Should catch unexpected exceptions and return False."""
        from musicmixer.services.sms import send_remix_ready

        mock_twilio_client.messages.create.side_effect = ConnectionError("network down")

        with caplog.at_level(logging.ERROR):
            result = send_remix_ready("+15551234567", "abc-123")

        assert result is False
        assert "Unexpected error sending remix-ready SMS" in caplog.text


class TestSendConfirmation:
    def test_sends_correct_message(self, sms_settings, mock_twilio_client):
        """Should send confirmation SMS with correct body and recipient."""
        from musicmixer.services.sms import send_confirmation

        result = send_confirmation("+15551234567")

        assert result is True
        mock_twilio_client.messages.create.assert_called_once_with(
            body="musicMixer: Got it! We'll text you when your remix is ready.",
            from_="+15550001111",
            to="+15551234567",
        )

    def test_returns_false_when_disabled(self, sms_settings, mock_twilio_client):
        """Should not send SMS when sms_enabled is False."""
        from musicmixer.services.sms import send_confirmation

        sms_settings.sms_enabled = False
        result = send_confirmation("+15551234567")

        assert result is False
        mock_twilio_client.messages.create.assert_not_called()

    def test_catches_twilio_exception(self, sms_settings, mock_twilio_client, caplog):
        """Should catch TwilioRestException and return False."""
        from musicmixer.services.sms import send_confirmation

        mock_twilio_client.messages.create.side_effect = TwilioRestException(
            status=400, uri="/Messages", msg="Invalid phone"
        )

        with caplog.at_level(logging.ERROR):
            result = send_confirmation("+15551234567")

        assert result is False
        assert "Failed to send confirmation SMS" in caplog.text

    def test_catches_generic_exception(self, sms_settings, mock_twilio_client, caplog):
        """Should catch unexpected exceptions and return False."""
        from musicmixer.services.sms import send_confirmation

        mock_twilio_client.messages.create.side_effect = RuntimeError("boom")

        with caplog.at_level(logging.ERROR):
            result = send_confirmation("+15551234567")

        assert result is False
        assert "Unexpected error sending confirmation SMS" in caplog.text
