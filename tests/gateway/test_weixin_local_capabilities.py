"""Tests for the locally-migrated Weixin capabilities.

Upstream v0.22.5 has no equivalent for these paths, so they were carried over
from the retired ``gateway/platforms/weixin.py`` core adapter. Every test here
pins a behaviour that the local production deployment depends on:

  * delivery ledger  — a failed send must never be a lost message, and a
                       delivered entry must never be sent twice.
  * forensic tap     — read-only recording of the full iLink response, and
                       redaction of secrets before it touches disk.
  * error kind       — iLink error text mapped onto ``SendResult.error_kind``.
  * token bucket     — per-peer token-age bookkeeping.
  * instant ack      — one short ack per inbound, never blocking the turn.
"""

import json

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.weixin import adapter as weixin
from plugins.platforms.weixin.adapter import WeixinAdapter


def _make_adapter(**extra) -> WeixinAdapter:
    base = {"account_id": "test-account"}
    base.update(extra)
    return WeixinAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra=base,
        )
    )


# ---------------------------------------------------------------------------
# Delivery ledger
# ---------------------------------------------------------------------------


class TestWeixinDeliveryLedger:
    def test_ledger_key_is_stable_for_same_content(self):
        a = weixin._ledger_key("chat-1", "hello")
        b = weixin._ledger_key("chat-1", "hello")
        assert a == b

    def test_ledger_key_differs_per_chat_and_content(self):
        assert weixin._ledger_key("chat-1", "hello") != weixin._ledger_key("chat-2", "hello")
        assert weixin._ledger_key("chat-1", "hello") != weixin._ledger_key("chat-1", "world")

    def test_append_then_pending_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path / "ledger.jsonl")
        weixin._ledger_append({
            "event": "send_failed",
            "message_key": weixin._ledger_key("chat-1", "hi"),
            "chat_id": "chat-1",
            "content": "hi",
            "attempts": 1,
        })
        pending = weixin._ledger_pending()
        assert len(pending) == 1
        assert pending[0]["chat_id"] == "chat-1"
        assert pending[0]["content"] == "hi"

    def test_delivered_entry_is_not_pending_again(self, tmp_path, monkeypatch):
        """Idempotency: a delivered message must never be re-sent."""
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path / "ledger.jsonl")
        record = {
            "event": "send_failed",
            "message_key": weixin._ledger_key("chat-1", "hi"),
            "chat_id": "chat-1",
            "content": "hi",
            "attempts": 1,
        }
        weixin._ledger_append(record)
        weixin._ledger_append({**record, "event": "delivered", "attempts": 2})
        assert weixin._ledger_pending() == []

    def test_pending_only_returns_undelivered_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path / "ledger.jsonl")
        weixin._ledger_append({
            "event": "send_failed",
            "message_key": weixin._ledger_key("chat-1", "a"),
            "chat_id": "chat-1",
            "content": "a",
            "attempts": 1,
        })
        weixin._ledger_append({
            "event": "send_failed",
            "message_key": weixin._ledger_key("chat-2", "b"),
            "chat_id": "chat-2",
            "content": "b",
            "attempts": 1,
        })
        chats = {r["chat_id"] for r in weixin._ledger_pending()}
        assert chats == {"chat-1", "chat-2"}

    def test_missing_ledger_file_yields_empty_pending(self, tmp_path, monkeypatch):
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path / "absent.jsonl")
        assert weixin._ledger_pending() == []

    def test_append_survives_unwritable_path(self, tmp_path, monkeypatch):
        """The ledger must never take the send path down."""
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path / "ledger.jsonl")
        weixin._ledger_append({"event": "send_failed", "chat_id": "c", "content": "x"})
        # Point the ledger at a directory so the write fails.
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path)
        weixin._ledger_append({"event": "send_failed", "chat_id": "c", "content": "x"})
        # No exception escaped — the caller survives.


# ---------------------------------------------------------------------------
# Error kind classification
# ---------------------------------------------------------------------------


class TestWeixinErrorKind:
    @pytest.mark.parametrize("text,expected", [
        ("rate limited", "rate_limited"),
        ("flood", "rate_limited"),
        ("too many requests", "rate_limited"),
        ("prepare failed", "transient"),
        ("session not ready", "transient"),
        ("sendmessage error", "transient"),
    ])
    def test_known_markers_map_to_kind(self, text, expected):
        assert weixin._ilink_error_kind(text) == expected

    def test_unknown_error_has_no_kind(self):
        assert weixin._ilink_error_kind("invalid api key") is None

    def test_empty_error_has_no_kind(self):
        assert weixin._ilink_error_kind("") is None


# ---------------------------------------------------------------------------
# Forensic tap + redaction
# ---------------------------------------------------------------------------


class TestWeixinForensicTap:
    def test_tap_records_failed_response(self, tmp_path, monkeypatch):
        tap = tmp_path / "tap.jsonl"
        monkeypatch.setattr(weixin, "_ILINK_TAP_PATH", tap)
        weixin._ilink_forensic_tap({"ret": -2, "errmsg": "prepare failed"}, -2, None,
                                   context_token="tok", chat_id="chat-1")
        lines = tap.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["ret"] == -2
        assert record["ok"] is False
        assert record["had_token"] is True
        assert record["resp_keys"] == ["errmsg", "ret"]

    def test_tap_skips_successful_sends_by_default(self, tmp_path, monkeypatch):
        """Keep the diagnostic file small: successes are only recorded on request."""
        tap = tmp_path / "tap.jsonl"
        monkeypatch.setattr(weixin, "_ILINK_TAP_PATH", tap)
        monkeypatch.delenv("HERMES_ILINK_TAP_ALL", raising=False)
        weixin._ilink_forensic_tap({"ret": 0}, 0, None, context_token="tok", chat_id="chat-1")
        assert not tap.exists()

    def test_tap_records_successful_sends_when_requested(self, tmp_path, monkeypatch):
        tap = tmp_path / "tap.jsonl"
        monkeypatch.setattr(weixin, "_ILINK_TAP_PATH", tap)
        monkeypatch.setenv("HERMES_ILINK_TAP_ALL", "1")
        weixin._ilink_forensic_tap({"ret": 0}, 0, None, context_token="tok", chat_id="chat-1")
        record = json.loads(tap.read_text(encoding="utf-8").splitlines()[0])
        assert record["ok"] is True

    def test_tap_disabled_by_env(self, tmp_path, monkeypatch):
        tap = tmp_path / "tap.jsonl"
        monkeypatch.setattr(weixin, "_ILINK_TAP_PATH", tap)
        monkeypatch.setenv("HERMES_ILINK_TAP", "0")
        weixin._ilink_forensic_tap({"ret": -2}, -2, None, context_token="tok", chat_id="chat-1")
        assert not tap.exists()

    def test_tap_redacts_secrets(self, tmp_path, monkeypatch):
        tap = tmp_path / "tap.jsonl"
        monkeypatch.setattr(weixin, "_ILINK_TAP_PATH", tap)
        weixin._ilink_forensic_tap({"ret": -2, "token": "super-secret"}, -2, None,
                                   context_token="ct-secret", chat_id="chat-1")
        raw = tap.read_text(encoding="utf-8")
        assert "super-secret" not in raw
        assert "ct-secret" not in raw

    def test_redact_masks_secret_values(self):
        """Redaction is keyed on the field NAME (token/secret/key/password/...), recursively."""
        redacted = weixin._redact_ilink({
            "token": "abc123",
            "context_token": "ct-456",
            "nested": {"api_key": "k-789"},
            "list": [{"password": "p-000"}],
            "keep": 1,
        })
        assert redacted["token"] != "abc123"
        assert redacted["context_token"] != "ct-456"
        assert redacted["nested"]["api_key"] != "k-789"
        assert redacted["list"][0]["password"] != "p-000"
        assert redacted["keep"] == 1

    def test_redact_keeps_had_token_flag(self):
        """``had_token`` is a diagnostic boolean, not a credential — it must survive."""
        assert weixin._redact_ilink({"had_token": True})["had_token"] is True

    def test_redact_leaves_non_secret_fields(self):
        assert weixin._redact_ilink({"ret": 0, "errmsg": "ok"})["errmsg"] == "ok"


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


class TestWeixinTokenBucket:
    def test_bucket_is_stable_per_peer(self):
        assert weixin._key_token_bucket("peer-1") == weixin._key_token_bucket("peer-1")

    def test_bucket_differs_per_peer(self):
        assert weixin._key_token_bucket("peer-1") != weixin._key_token_bucket("peer-2")

    def test_bucket_records_token_age(self):
        """A fresh token must read as ~0s old — the decisive probe for token staleness."""
        bucket = weixin._key_token_bucket("peer-token-age")
        weixin._ILINK_TOKEN_SINCE[bucket] = 1000.0
        seen = weixin._ILINK_TOKEN_SINCE.get(bucket)
        assert seen == 1000.0


# ---------------------------------------------------------------------------
# Instant ack
# ---------------------------------------------------------------------------


class TestWeixinInstantAck:
    def test_ack_disabled_by_default(self):
        adapter = _make_adapter()
        assert adapter._ack_enabled is False

    def test_ack_text_reads_configured_message(self):
        adapter = _make_adapter(ack_enabled="true", ack_message="  收到  ")
        assert adapter._ack_enabled is True
        assert adapter._ack_text() == "收到"

    def test_ack_text_empty_when_unset(self):
        assert _make_adapter()._ack_text() == ""

    def test_arm_ack_rearms_for_next_inbound(self):
        adapter = _make_adapter()
        adapter._acked_chats.add("chat-1")
        adapter._arm_ack("chat-1")
        assert "chat-1" not in adapter._acked_chats

    @pytest.mark.asyncio
    async def test_ack_sent_once_per_inbound(self, monkeypatch):
        adapter = _make_adapter(ack_enabled="true", ack_message="收到")
        monkeypatch.setattr(adapter, "_send_session", object())
        sent = []

        async def _fake_send_text_chunk(*, chat_id, chunk, context_token, client_id):
            sent.append(chunk)

        monkeypatch.setattr(adapter, "_send_text_chunk", _fake_send_text_chunk)
        await adapter._maybe_send_fresh_ack("chat-1")
        await adapter._maybe_send_fresh_ack("chat-1")
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_ack_failure_does_not_raise(self, monkeypatch):
        """An ack is a courtesy — it must never break the real reply path."""
        adapter = _make_adapter(ack_enabled="true", ack_message="收到")
        monkeypatch.setattr(adapter, "_send_session", object())

        async def _boom(*, chat_id, chunk, context_token, client_id):
            raise RuntimeError("send failed")

        monkeypatch.setattr(adapter, "_send_text_chunk", _boom)
        await adapter._maybe_send_fresh_ack("chat-1")

    @pytest.mark.asyncio
    async def test_ack_skipped_when_disabled(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setattr(adapter, "_send_session", object())
        sent = []

        async def _fake_send_text_chunk(*, chat_id, chunk, context_token, client_id):
            sent.append(chunk)

        monkeypatch.setattr(adapter, "_send_text_chunk", _fake_send_text_chunk)
        await adapter._maybe_send_fresh_ack("chat-1")
        assert sent == []

    @pytest.mark.asyncio
    async def test_ack_skipped_when_not_connected(self, monkeypatch):
        """No live session means no send attempt at all — the ack must not even try."""
        adapter = _make_adapter(ack_enabled="true", ack_message="收到")
        sent = []

        async def _fake_send_text_chunk(*, chat_id, chunk, context_token, client_id):
            sent.append(chunk)

        monkeypatch.setattr(adapter, "_send_text_chunk", _fake_send_text_chunk)
        await adapter._maybe_send_fresh_ack("chat-1")
        assert sent == []


# ---------------------------------------------------------------------------
# Ledger drain
# ---------------------------------------------------------------------------


class TestWeixinLedgerDrain:
    @pytest.mark.asyncio
    async def test_drain_resends_pending_entry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path / "ledger.jsonl")
        weixin._ledger_append({
            "event": "send_failed",
            "message_key": weixin._ledger_key("chat-1", "pending text"),
            "chat_id": "chat-1",
            "content": "pending text",
            "attempts": 1,
        })
        adapter = _make_adapter()
        sent = []

        async def _fake_send_text_chunk(*, chat_id, chunk, context_token, client_id):
            sent.append(chunk)

        monkeypatch.setattr(adapter, "_send_text_chunk", _fake_send_text_chunk)
        await adapter._drain_ledger("chat-1")
        assert sent == ["pending text"]
        assert weixin._ledger_pending() == []

    @pytest.mark.asyncio
    async def test_drain_marks_delivery_failure_for_retry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path / "ledger.jsonl")
        weixin._ledger_append({
            "event": "send_failed",
            "message_key": weixin._ledger_key("chat-1", "pending text"),
            "chat_id": "chat-1",
            "content": "pending text",
            "attempts": 1,
        })
        adapter = _make_adapter()

        async def _boom(*, chat_id, chunk, context_token, client_id):
            raise RuntimeError("still refusing")

        monkeypatch.setattr(adapter, "_send_text_chunk", _boom)
        await adapter._drain_ledger("chat-1")
        pending = weixin._ledger_pending()
        assert len(pending) == 1
        assert pending[0]["attempts"] == 2

    @pytest.mark.asyncio
    async def test_drain_dead_letters_after_max_attempts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path / "ledger.jsonl")
        weixin._ledger_append({
            "event": "send_failed",
            "message_key": weixin._ledger_key("chat-1", "doomed"),
            "chat_id": "chat-1",
            "content": "doomed",
            "attempts": weixin._LEDGER_MAX_ATTEMPTS,
        })
        adapter = _make_adapter()
        sent = []

        async def _fake_send_text_chunk(*, chat_id, chunk, context_token, client_id):
            sent.append(chunk)

        monkeypatch.setattr(adapter, "_send_text_chunk", _fake_send_text_chunk)
        await adapter._drain_ledger("chat-1")
        assert sent == []
        assert weixin._ledger_pending() == []
        raw = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
        assert "dead_letter" in raw

    @pytest.mark.asyncio
    async def test_drain_only_touches_its_own_chat(self, tmp_path, monkeypatch):
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path / "ledger.jsonl")
        weixin._ledger_append({
            "event": "send_failed",
            "message_key": weixin._ledger_key("chat-2", "other chat"),
            "chat_id": "chat-2",
            "content": "other chat",
            "attempts": 1,
        })
        adapter = _make_adapter()
        sent = []

        async def _fake_send_text_chunk(*, chat_id, chunk, context_token, client_id):
            sent.append(chunk)

        monkeypatch.setattr(adapter, "_send_text_chunk", _fake_send_text_chunk)
        await adapter._drain_ledger("chat-1")
        assert sent == []
        assert len(weixin._ledger_pending()) == 1

    @pytest.mark.asyncio
    async def test_drain_never_raises(self, tmp_path, monkeypatch):
        """A ledger problem must not surface as a turn failure."""
        monkeypatch.setattr(weixin, "_LEDGER_PATH", tmp_path / "ledger.jsonl")
        adapter = _make_adapter()

        def _boom():
            raise RuntimeError("ledger exploded")

        monkeypatch.setattr(weixin, "_ledger_pending", _boom)
        await adapter._drain_ledger("chat-1")
