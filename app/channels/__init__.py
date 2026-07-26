"""Seleção do provedor de canal."""

from app.channels.base import Channel, IncomingMessage, split_message
from app.channels.meta_cloud import MetaCloudChannel
from app.channels.twilio import TwilioChannel
from app.config import PROVIDER_TWILIO, Settings

__all__ = [
    "Channel",
    "IncomingMessage",
    "MetaCloudChannel",
    "TwilioChannel",
    "build_channel",
    "split_message",
]


def build_channel(settings: Settings) -> Channel:
    if settings.provider == PROVIDER_TWILIO:
        return TwilioChannel(settings)
    return MetaCloudChannel(settings)
