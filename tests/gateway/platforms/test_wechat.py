"""Tests for the WeChat gateway adapter."""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _make_adapter(**extra):
    """Create a WeChatAdapter with test defaults."""
    from gateway.platforms.wechat import WeChatAdapter

    config = PlatformConfig(enabled=True, extra=extra)
    adapter = WeChatAdapter(config)
    adapter._running = True
    adapter._http_session = MagicMock()
    return adapter


def test_send_text_happy_path():
    adapter = _make_adapter(bridge_host="127.0.0.1", bridge_port=18400)

    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True, "messageId": "msg-1"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

    result = asyncio.run(adapter.send("wxid_123", "hello from hermes"))

    assert result.success is True
    assert result.message_id == "msg-1"
    payload = adapter._http_session.post.call_args.kwargs["json"]
    assert payload == {"chatId": "wxid_123", "message": "hello from hermes"}


def test_send_401_logs_clear_error_without_retry(caplog):
    adapter = _make_adapter()

    resp = MagicMock(status=401)
    resp.text = AsyncMock(return_value='{"success":false,"error":"auth_expired"}')
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

    with caplog.at_level("ERROR"):
        result = asyncio.run(adapter.send("wxid_123", "hello"))

    assert result.success is False
    assert adapter._http_session.post.call_count == 1
    assert "auth/subscription expired" in caplog.text
    assert "re-activate" in caplog.text


def test_send_reply_to_degrades_silently_without_replyto():
    adapter = _make_adapter()

    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True, "messageId": "msg-2"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

    result = asyncio.run(adapter.send("wxid_123", "reply fallback", reply_to="orig-1"))

    assert result.success is True
    payload = adapter._http_session.post.call_args.kwargs["json"]
    assert "replyTo" not in payload


def test_health_poll_marks_connected_then_degraded():
    adapter = _make_adapter()

    connected = MagicMock(status=200)
    connected.json = AsyncMock(return_value={"status": "connected", "queueLength": 0, "uptime": 1.0})
    degraded = MagicMock(status=200)
    degraded.json = AsyncMock(return_value={"status": "degraded", "queueLength": 7, "uptime": 2.0})

    adapter._http_session.get = MagicMock(side_effect=[_AsyncCM(connected), _AsyncCM(degraded)])
    adapter._force_reconnect = MagicMock()

    with patch("gateway.status.write_runtime_status") as mock_write_status:
        ok = asyncio.run(adapter._check_health_once())
        not_ok = asyncio.run(adapter._check_health_once())

    assert ok is True
    assert not_ok is False
    adapter._force_reconnect.assert_called_once()
    degraded_calls = [call.kwargs for call in mock_write_status.call_args_list if call.kwargs.get("platform_state") == "degraded"]
    assert degraded_calls


def test_sse_event_translates_to_message_event():
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()

    sample = {
        "messageId": "mid-123",
        "chatId": "wxid_group@chatroom",
        "senderId": "wxid_sender",
        "senderName": "Alice",
        "chatName": "Project Group",
        "isGroup": True,
        "body": "hello from wechat",
        "hasMedia": False,
        "mediaType": "",
        "mediaUrls": [],
        "mentionedIds": [],
        "quotedParticipant": None,
        "botIds": [],
        "timestamp": 1713859200,
    }

    asyncio.run(adapter._handle_stream_payload(sample))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello from wechat"
    assert event.message_id == "mid-123"
    assert event.source.platform == Platform.WECHAT
    assert event.source.chat_id == "wxid_group@chatroom"
    assert event.source.user_id == "wxid_sender"
    assert event.source.chat_type == "group"


def test_get_chat_history_returns_list_payload():
    adapter = _make_adapter()

    history = [
        {"messageId": "m1", "body": "hello"},
        {"messageId": "m2", "body": "world"},
    ]
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value=history)
    adapter._http_session.get = MagicMock(return_value=_AsyncCM(resp))

    result = asyncio.run(adapter.get_chat_history("wxid_123", limit=2))

    assert result == history


def test_get_chat_history_accepts_wrapped_list_payload():
    adapter = _make_adapter()

    history = [{"messageId": "m1", "body": "hello"}]
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"data": history})
    adapter._http_session.get = MagicMock(return_value=_AsyncCM(resp))

    result = asyncio.run(adapter.get_chat_history("wxid_123", limit=1))

    assert result == history


def test_consume_sse_response_accepts_list_payloads():
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    adapter._running = True

    payload = (
        'data: [{"messageId":"mid-1","chatId":"filehelper","senderId":"wxid_sender",'
        '"senderName":"Alice","chatName":"File Helper","isGroup":false,"body":"hello",'
        '"hasMedia":false,"mediaType":"","mediaUrls":[],"mentionedIds":[],'
        '"quotedParticipant":null,"botIds":[],"timestamp":1713859200}]\n'
    ).encode()

    class _Content:
        async def iter_chunked(self, _size):
            yield payload

    response = MagicMock()
    response.content = _Content()

    asyncio.run(adapter._consume_sse_response(response))

    adapter.handle_message.assert_awaited_once()


def test_connect_releases_bridge_lock_on_failed_startup():
    from gateway.platforms.wechat import WeChatAdapter

    adapter = WeChatAdapter(PlatformConfig(enabled=True))

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    fake_aiohttp = types.SimpleNamespace(ClientSession=MagicMock(return_value=session))

    with patch("gateway.platforms.wechat.check_wechat_requirements", return_value=True), \
         patch.dict("sys.modules", {"aiohttp": fake_aiohttp}), \
         patch.object(adapter, "_acquire_platform_lock", return_value=True) as acquire_lock, \
         patch.object(adapter, "_release_platform_lock") as release_lock, \
         patch.object(adapter, "_check_health_once", new=AsyncMock(return_value=False)):
        ok = asyncio.run(adapter.connect())

    assert ok is False
    acquire_lock.assert_called_once()
    release_lock.assert_called_once()
