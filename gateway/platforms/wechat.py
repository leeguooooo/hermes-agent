"""WeChat platform adapter backed by a local HTTP/SSE bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import quote

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

DEFAULT_BRIDGE_HOST = "127.0.0.1"
DEFAULT_BRIDGE_PORT = 18400
MAX_MESSAGE_LENGTH = 4096
SSE_RETRY_DELAY_INITIAL = 2.0
SSE_RETRY_DELAY_MAX = 30.0
HEALTH_CHECK_INTERVAL = 20.0
SEND_RETRYABLE_STATUSES = {503}
HTTP_CONNECT_TIMEOUT = 10.0
HTTP_READ_TIMEOUT = 30.0
SSE_READ_TIMEOUT = 75.0


def check_wechat_requirements() -> bool:
    """Check whether the runtime has the dependency needed for the bridge."""
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return True


class WeChatAdapter(BasePlatformAdapter):
    """Hermes gateway adapter for the local `wechat-bridge` process."""

    platform = Platform.WECHAT
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WECHAT)

        extra = config.extra or {}
        self._bridge_host = str(
            extra.get("bridge_host")
            or os.getenv("WECHAT_BRIDGE_HOST", DEFAULT_BRIDGE_HOST)
        ).strip() or DEFAULT_BRIDGE_HOST
        self._bridge_port = int(
            extra.get("bridge_port")
            or os.getenv("WECHAT_BRIDGE_PORT", str(DEFAULT_BRIDGE_PORT))
        )
        self._bridge_bearer = str(
            extra.get("bridge_bearer")
            or extra.get("bearer")
            or os.getenv("WECHAT_BRIDGE_BEARER", "")
        ).strip() or None

        self._base_url = f"http://{self._bridge_host}:{self._bridge_port}"
        self._http_session = None
        self._stream_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._stream_response = None
        self._cursor = max(int(time.time()) - 5, 0)
        self._recent_message_ids: deque[str] = deque(maxlen=256)
        self._recent_message_id_set: set[str] = set()
        self._last_health_status: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to the local bridge and start stream + health tasks."""
        if not check_wechat_requirements():
            logger.warning("[%s] aiohttp not installed. Run: pip install 'hermes-agent[messaging]'", self.name)
            return False

        import aiohttp

        lock_acquired = False
        started = False
        try:
            if not self._acquire_platform_lock("wechat-bridge", self._base_url, "WeChat bridge"):
                return False
            lock_acquired = True
        except Exception as exc:
            logger.warning("[%s] Could not acquire bridge lock (non-fatal): %s", self.name, exc)

        try:
            if self._http_session and not self._http_session.closed:
                await self._http_session.close()

            self._http_session = aiohttp.ClientSession()
            self._running = True

            health_ok = await self._check_health_once()
            if self.has_fatal_error:
                if self._http_session and not self._http_session.closed:
                    await self._http_session.close()
                self._http_session = None
                return False

            if not health_ok and self._last_health_status is None:
                logger.warning("[%s] WeChat bridge is unreachable at startup", self.name)
                self._running = False
                if self._http_session and not self._http_session.closed:
                    await self._http_session.close()
                self._http_session = None
                return False

            self._stream_task = asyncio.create_task(self._stream_messages(), name="wechat-stream")
            self._health_task = asyncio.create_task(self._health_monitor(), name="wechat-health")
            started = True
            return True
        finally:
            if not started and lock_acquired:
                self._release_platform_lock()

    async def disconnect(self) -> None:
        """Stop background tasks and close the HTTP session."""
        self._running = False

        for task in (self._stream_task, self._health_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._stream_task = None
        self._health_task = None

        if self._stream_response is not None:
            try:
                self._stream_response.close()
            except Exception:
                pass
            self._stream_response = None

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None

        self._release_platform_lock()
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message to the bridge."""
        del metadata
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)

        last_message_id = None
        for chunk in self.truncate_message(content, self.MAX_MESSAGE_LENGTH):
            payload = {
                "chatId": chat_id,
                "message": chunk,
            }
            status, data = await self._request_json(
                "POST",
                "/send",
                json_body=payload,
                timeout=30,
                retry_on_503=True,
            )

            if status in (401, 402):
                message = self._auth_reactivation_message()
                logger.error("[%s] WeChat send failed: %s", self.name, message)
                return SendResult(success=False, error=message)
            if status == 400 and data.get("error") == "reply_not_supported":
                logger.debug("[%s] WeChat bridge does not support replyTo; degraded to plain send", self.name)
                return SendResult(success=True, message_id=last_message_id)
            if status == 501:
                logger.warning("[%s] WeChat bridge returned 501 for /send", self.name)
                return SendResult(success=False, error="WeChat bridge does not support this send operation (501)")
            if status != 200:
                error = self._response_error(data, f"WeChat bridge error ({status})")
                return SendResult(success=False, error=error)
            if data and not data.get("success", True):
                error = self._response_error(data, "WeChat bridge send failed")
                return SendResult(success=False, error=error)
            last_message_id = data.get("messageId")

        if reply_to:
            logger.debug("[%s] WeChat reply_to is not supported yet; sent plain text instead", self.name)

        return SendResult(success=True, message_id=last_message_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Typing is intentionally a no-op for bridge v1.10.2."""
        del chat_id, metadata
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Fetch chat metadata from the bridge."""
        if not self._http_session:
            return {"name": chat_id, "type": "dm", "chat_id": chat_id}

        status, data = await self._request_json(
            "GET",
            f"/chat/{quote(str(chat_id), safe='')}",
            timeout=10,
        )
        if status == 200:
            return {
                "name": data.get("name", chat_id),
                "type": "group" if data.get("isGroup") else "dm",
                "chat_id": chat_id,
                "participants": data.get("participants", []),
            }
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    async def get_chat_history(
        self,
        chat_id: str,
        *,
        limit: int = 20,
        since: Optional[int] = None,
        until: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Best-effort recent message fetch from the bridge."""
        if not self._http_session:
            return []

        params = [f"limit={max(int(limit), 1)}"]
        if since is not None:
            params.append(f"since={int(since)}")
        if until is not None:
            params.append(f"until={int(until)}")
        query = "&".join(params)
        status, data = await self._request_json(
            "GET",
            f"/chat/{quote(str(chat_id), safe='')}/history?{query}",
            timeout=15,
        )
        if status == 200:
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                return data["data"]
        return []

    async def _health_monitor(self) -> None:
        """Continuously poll the bridge health endpoint."""
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break
            try:
                await self._check_health_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[%s] WeChat health monitor error: %s", self.name, exc)

    async def _check_health_once(self) -> bool:
        """Poll `/health` once and update runtime state."""
        if not self._http_session:
            return False

        status, data = await self._request_json("GET", "/health", timeout=10)
        if status in (401, 402):
            message = self._auth_reactivation_message()
            logger.error("[%s] WeChat health check failed: %s", self.name, message)
            self._set_fatal_error("wechat_bridge_auth", message, retryable=False)
            return False
        if status == 503:
            self._set_degraded("WeChat bridge is temporarily unavailable (503)")
            self._force_reconnect()
            return False
        if status != 200:
            self._set_degraded(f"WeChat bridge health check failed ({status})")
            self._force_reconnect()
            return False

        bridge_status = str(data.get("status") or "unknown").strip().lower()
        self._last_health_status = bridge_status
        if bridge_status == "connected":
            self._mark_connected()
            return True

        self._set_degraded(f"WeChat bridge status is {bridge_status or 'unknown'}")
        self._force_reconnect()
        return False

    def _set_degraded(self, message: str) -> None:
        """Write a degraded runtime state without marking the adapter fatal."""
        self._running = True
        try:
            from gateway.status import write_runtime_status

            write_runtime_status(
                platform=self.platform.value,
                platform_state="degraded",
                error_code="wechat_bridge_degraded",
                error_message=message,
            )
        except Exception:
            pass

    def _force_reconnect(self) -> None:
        """Close the active SSE response so the stream task reconnects."""
        response = self._stream_response
        if response is None:
            return
        self._stream_response = None
        try:
            response.close()
        except Exception:
            try:
                response.release()
            except Exception:
                pass

    async def _stream_messages(self) -> None:
        """Consume the bridge SSE endpoint and forward messages into Hermes."""
        if not self._http_session:
            return

        import aiohttp

        backoff = SSE_RETRY_DELAY_INITIAL
        while self._running:
            url = f"{self._base_url}/messages/stream?since={self._cursor}"
            try:
                async with self._http_session.get(
                    url,
                    headers=self._request_headers(),
                    timeout=aiohttp.ClientTimeout(
                        total=None,
                        connect=HTTP_CONNECT_TIMEOUT,
                        sock_connect=HTTP_CONNECT_TIMEOUT,
                        sock_read=SSE_READ_TIMEOUT,
                    ),
                ) as response:
                    self._stream_response = response
                    if response.status in (401, 402):
                        message = self._auth_reactivation_message()
                        logger.error("[%s] WeChat SSE failed: %s", self.name, message)
                        self._set_fatal_error("wechat_bridge_auth", message, retryable=False)
                        break
                    if response.status == 503:
                        logger.warning("[%s] WeChat SSE unavailable (503), retrying", self.name)
                        self._set_degraded("WeChat bridge stream is temporarily unavailable")
                    elif response.status != 200:
                        logger.warning("[%s] WeChat SSE returned HTTP %s", self.name, response.status)
                        self._set_degraded(f"WeChat SSE returned HTTP {response.status}")
                    else:
                        backoff = SSE_RETRY_DELAY_INITIAL
                        await self._consume_sse_response(response)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._running:
                    logger.warning("[%s] WeChat SSE error: %s", self.name, exc)

            self._stream_response = None
            if not self._running:
                break
            await asyncio.sleep(backoff + (backoff * 0.2 * random.random()))
            backoff = min(backoff * 2, SSE_RETRY_DELAY_MAX)

    async def _consume_sse_response(self, response) -> None:
        """Parse an aiohttp SSE response and dispatch each `data:` payload."""
        buffer = ""
        async for raw_chunk in response.content.iter_chunked(4096):
            if not self._running:
                break
            buffer += raw_chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    logger.debug("[%s] Invalid WeChat SSE payload: %s", self.name, payload[:200])
                    continue
                if isinstance(data, list):
                    for item in data:
                        await self._handle_stream_payload(item)
                else:
                    await self._handle_stream_payload(data)

    async def _handle_stream_payload(self, data: Dict[str, Any]) -> None:
        """Translate one bridge payload into a Hermes `MessageEvent`."""
        message_id = str(data.get("messageId") or "").strip()
        if message_id:
            if message_id in self._recent_message_id_set:
                return
            if len(self._recent_message_ids) == self._recent_message_ids.maxlen:
                expired = self._recent_message_ids.popleft()
                self._recent_message_id_set.discard(expired)
            self._recent_message_ids.append(message_id)
            self._recent_message_id_set.add(message_id)

        self._cursor = max(self._cursor, self._cursor_from_timestamp(data.get("timestamp")))

        event = self._build_message_event(data)
        if event is None:
            return
        await self.handle_message(event)

    def _build_message_event(self, data: Dict[str, Any]) -> Optional[MessageEvent]:
        """Map a bridge payload to Hermes' normalized inbound event."""
        if not isinstance(data, dict):
            return None

        chat_id = str(data.get("chatId") or "").strip()
        sender_id = str(data.get("senderId") or "").strip()
        if not chat_id or not sender_id:
            return None

        bot_ids = {str(candidate).strip() for candidate in (data.get("botIds") or []) if str(candidate).strip()}
        if sender_id in bot_ids:
            return None

        media_urls = [str(url) for url in (data.get("mediaUrls") or []) if str(url).strip()]
        media_type = str(data.get("mediaType") or "").strip().lower()
        message_type = MessageType.TEXT
        if data.get("hasMedia"):
            if "image" in media_type:
                message_type = MessageType.PHOTO
            elif "video" in media_type:
                message_type = MessageType.VIDEO
            elif "audio" in media_type or "voice" in media_type or "ptt" in media_type:
                message_type = MessageType.VOICE
            else:
                message_type = MessageType.DOCUMENT

        chat_type = "group" if data.get("isGroup") else "dm"
        source = self.build_source(
            chat_id=chat_id,
            chat_name=data.get("chatName"),
            chat_type=chat_type,
            user_id=sender_id,
            user_name=data.get("senderName"),
        )

        return MessageEvent(
            text=str(data.get("body") or ""),
            message_type=message_type,
            source=source,
            raw_message=data,
            message_id=str(data.get("messageId") or "") or None,
            media_urls=media_urls,
            media_types=[media_type] * len(media_urls),
            timestamp=self._parse_timestamp(data.get("timestamp")),
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        retry_on_503: bool = False,
    ) -> tuple[int, Any]:
        """Issue a bridge request and preserve dict/list JSON payloads."""
        if not self._http_session:
            return 0, {"message": "HTTP session not initialized"}

        url = f"{self._base_url}{path}"
        attempt = 0
        backoff = 1.0
        request_fn = getattr(self._http_session, method.lower(), None)
        if request_fn is None:
            request_fn = lambda request_url, **kwargs: self._http_session.request(method, request_url, **kwargs)

        while True:
            request_kwargs = {
                "json": json_body,
                "headers": self._request_headers(),
            }
            timeout_value = self._client_timeout(timeout)
            if timeout_value is not None:
                request_kwargs["timeout"] = timeout_value

            async with request_fn(
                url,
                **request_kwargs,
            ) as response:
                payload = await self._coerce_json_payload(response)
                if response.status in SEND_RETRYABLE_STATUSES and retry_on_503 and attempt < 3:
                    logger.warning(
                        "[%s] WeChat bridge returned HTTP %d, retrying in %.1fs",
                        self.name,
                        response.status,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    attempt += 1
                    backoff = min(backoff * 2, 8.0)
                    continue
                return response.status, payload

    async def _coerce_json_payload(self, response) -> Any:
        """Decode a bridge response into dict/list/scalar JSON data."""
        if hasattr(response, "json"):
            try:
                data = await response.json()
                return data
            except Exception:
                pass

        text = ""
        if hasattr(response, "text"):
            text = await response.text()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"message": text}
        return data

    @staticmethod
    def _client_timeout(total: int):
        """Build an aiohttp timeout object when aiohttp is available."""
        try:
            import aiohttp
        except ImportError:
            return None
        total_timeout = float(total)
        connect_timeout = min(total_timeout, HTTP_CONNECT_TIMEOUT)
        read_timeout = min(total_timeout, HTTP_READ_TIMEOUT)
        return aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_connect=connect_timeout,
            sock_read=read_timeout,
        )

    def _request_headers(self) -> Dict[str, str]:
        """Common unary + SSE request headers."""
        headers = {"Accept": "application/json, text/event-stream"}
        if self._bridge_bearer:
            headers["Authorization"] = f"Bearer {self._bridge_bearer}"
        return headers

    def _auth_reactivation_message(self) -> str:
        return "auth/subscription expired — user must re-activate"

    @staticmethod
    def _response_error(data: Dict[str, Any], fallback: str) -> str:
        message = str(data.get("message") or data.get("error") or fallback)
        error_code = str(data.get("error") or "").strip()
        if error_code and error_code not in message:
            return f"{fallback}: {error_code}"
        return message

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        """Convert bridge timestamps to timezone-aware datetimes."""
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return datetime.now(tz=timezone.utc)
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                pass
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return datetime.now(tz=timezone.utc)
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _cursor_from_timestamp(value: Any) -> int:
        """Normalize a bridge timestamp into the integer `since=` cursor."""
        if isinstance(value, (int, float)):
            try:
                return max(int(float(value)), 0)
            except (TypeError, ValueError):
                return max(int(time.time()) - 5, 0)
        if isinstance(value, str) and value.strip():
            try:
                return max(int(float(value)), 0)
            except (TypeError, ValueError):
                return max(int(time.time()) - 5, 0)
        return max(int(time.time()) - 5, 0)
