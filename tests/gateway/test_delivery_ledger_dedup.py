"""Regression tests for the delivery ledger's dead-row verdicts and duplicate suppression.

Three defects are reproduced here. Each test asserts the CORRECT behaviour, so it fails
before the fix and passes after it:

- A1 ``sweep_recoverable`` skips every row whose owner is a LIVE process, so a row this
  process itself failed can never reach the attempts-cap verdict and stays ``failed`` for
  good. The stale-session budget is then spent forever with no verdict.
- A2 ``sweep_failed_for_runtime`` asks ``retry_not_before()`` first. For a stale-session row
  whose budget is spent that returns ``None``, and the ``if due is None: continue`` guard
  skips the attempts-cap verdict that follows it — so the row is never abandoned either.
  The comment says "the sweep turns it into abandoned", but the code never reaches it.
- B  ``compute_obligation_id`` hashes the raw content, so the SAME answer recorded with a
  recovered/flood marker (or a plain-text downgrade) becomes a SECOND obligation row, and
  the duplicate-suppression the ledger is supposed to provide silently does not happen.

The crash-recovery path (``record_crash_left_reply``) has the same duplicate problem: its
"not already ledgered" guard compares the raw content, so a marker variant slips past it.
"""

import time

import pytest

from gateway import delivery_ledger as dl

STALE_ERROR = ("iLink sendmessage session not ready: ret=-2 errcode=None "
               "errmsg=prepare failed")


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Isolated state.db per test; the production ledger is never touched."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _record(oid="ob-1", session_key="agent:weixin:weixin:dm:X", platform="weixin",
            chat_id="C1", content="the final answer", thread_id=None):
    dl.record_obligation(
        obligation_id=oid,
        session_key=session_key,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        content=content,
    )


def _row(oid):
    with dl._connect() as conn:
        r = conn.execute(
            "SELECT state, attempts, owner_pid, content, last_error "
            "FROM delivery_obligations WHERE obligation_id=?", (oid,)
        ).fetchone()
    return None if r is None else {
        "state": r[0], "attempts": r[1], "owner_pid": r[2],
        "content": r[3], "last_error": r[4],
    }


def _count():
    with dl._connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()[0]


def _set_state(oid, state, attempts, error, owner_pid=None, owner_started_at=None):
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET state=?, attempts=?, last_error=?, "
            "owner_pid=COALESCE(?, owner_pid), owner_started_at=COALESCE(?, owner_started_at) "
            "WHERE obligation_id=?", (state, attempts, error, owner_pid, owner_started_at, oid)
        )


# --------------------------------------------------------------------------- A1

def test_failed_row_owned_by_live_process_is_abandoned_at_cap(monkeypatch):
    """A1: a row this process failed must still get a verdict once its budget is spent.

    The owner is alive (it is THIS process), so the owner check must not stop the
    attempts-cap verdict from running: a 'failed' row has no delivery in flight, and
    leaving it 'failed' forever is what lets the stale-session budget spin.
    """
    _record("ob-live")
    _set_state("ob-live", "failed", dl.STALE_SESSION_ATTEMPTS, STALE_ERROR)
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: True)

    claimed = dl.sweep_recoverable(deliverable_platforms={"weixin"})

    assert _row("ob-live")["state"] == "abandoned", (
        "a failed row whose budget is spent must be abandoned (left for the record), "
        "not parked in 'failed' forever"
    )
    assert claimed == [], "an exhausted row must not be handed out for redelivery"


def test_failed_row_owned_by_live_process_with_budget_left_is_left_to_runtime_sweep(monkeypatch):
    """A failed row whose owner is alive and whose budget is left is NOT the boot sweep's
    business: the runtime sweep owns this process's own rows. The fix must leave it alone
    (not abandon it, not claim it) so the two sweeps keep their separate jobs."""
    _record("ob-live2")
    _set_state("ob-live2", "failed", 1, STALE_ERROR)
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: True)

    claimed = dl.sweep_recoverable(deliverable_platforms={"weixin"})

    assert claimed == [], "the boot sweep must not steal the runtime sweep's own row"
    assert _row("ob-live2")["state"] == "failed", "a row with budget left must not be abandoned"


def test_failed_row_with_dead_owner_and_budget_left_is_still_redelivered(monkeypatch):
    """The fix must not swallow messages: a row with budget left is still delivered."""
    _record("ob-live3")
    _set_state("ob-live3", "failed", 1, STALE_ERROR)
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: False)

    claimed = dl.sweep_recoverable(deliverable_platforms={"weixin"})

    assert [c["obligation_id"] for c in claimed] == ["ob-live3"]
    assert _row("ob-live3")["state"] == "attempting"


# --------------------------------------------------------------------------- A2

def test_runtime_sweep_leaves_the_last_attempt_to_the_boot_sweep():
    """The runtime sweep must NOT spend the last budgeted attempt, and must not judge the row
    either: ``retry_not_before`` returns None for a budget-spent stale row ON PURPOSE — the last
    attempt belongs to the boot sweep, the only path that can also abandon the row. Judging it
    here would take that last send away and silently drop the answer."""
    _record("ob-rt")
    pid, started = dl._owner_stamp()
    _set_state("ob-rt", "failed", dl.STALE_SESSION_ATTEMPTS, STALE_ERROR,
               owner_pid=pid, owner_started_at=started)

    claimed = dl.sweep_failed_for_runtime("weixin")

    assert claimed == [], "the runtime sweep must not spend the last budgeted attempt"
    assert _row("ob-rt")["state"] == "failed", (
        "the verdict belongs to the boot sweep, which can still deliver the last attempt"
    )


def test_runtime_sweep_still_redelivers_within_budget():
    """A row inside its budget is still due and must be handed back for redelivery."""
    _record("ob-rt2")
    pid, started = dl._owner_stamp()
    _set_state("ob-rt2", "failed", 1, STALE_ERROR, owner_pid=pid, owner_started_at=started)
    with dl._connect() as conn:  # push the backoff deadline into the past
        conn.execute("UPDATE delivery_obligations SET updated_at=? WHERE obligation_id=?",
                     (time.time() - 3600, "ob-rt2"))

    claimed = dl.sweep_failed_for_runtime("weixin")

    assert [c["obligation_id"] for c in claimed] == ["ob-rt2"]
    assert _row("ob-rt2")["state"] == "attempting"


# --------------------------------------------------------------------------- B

def test_same_answer_with_marker_shares_one_obligation():
    """B: the recovered/flood marker is presentation, not a different answer."""
    a = dl.compute_obligation_id("agent:weixin:weixin:dm:X", "m1", "the final answer")
    b = dl.compute_obligation_id("agent:weixin:weixin:dm:X", "m1",
                                  dl.RECOVERED_MARKER + "the final answer")
    c = dl.compute_obligation_id("agent:weixin:weixin:dm:X", "m1",
                                  dl.FLOOD_MARKER + "the final answer")
    assert a == b == c, (
        "a marker variant is the same answer: it must map to the same obligation id so "
        "the ledger keeps ONE row and the answer is delivered once"
    )


def test_marker_variant_records_into_the_same_row():
    """B: recording the marker variant must not create a second row."""
    _record("ob-dup", content="the final answer")
    _record("ob-dup", content=dl.RECOVERED_MARKER + "the final answer")
    assert _count() == 1, "the same answer must occupy exactly one ledger row"


def test_distinct_message_refs_are_not_merged():
    """B must not swallow a real second answer to the same question."""
    a = dl.compute_obligation_id("agent:weixin:weixin:dm:X", "m1", "same text")
    b = dl.compute_obligation_id("agent:weixin:weixin:dm:X", "m2", "same text")
    assert a != b, "two different inbound turns are two different obligations"


def test_crash_left_reply_does_not_duplicate_the_same_answer():
    """B: the crash-recovery guard compares raw content, so a marker variant slips past."""
    dl.record_crash_left_reply(obligation_id="c1", session_key="agent:weixin:weixin:dm:X",
                                platform="weixin", chat_id="C1", thread_id=None,
                                content="the final answer", since=time.time() - 60)
    dl.record_crash_left_reply(obligation_id="c2", session_key="agent:weixin:weixin:dm:X",
                                platform="weixin", chat_id="C1", thread_id=None,
                                content=dl.RECOVERED_MARKER + "the final answer",
                                since=time.time() - 60)
    assert _count() == 1, (
        "a recovered marker variant of an already-ledgered answer must not become a "
        "second row"
    )


def test_normalize_content_strips_every_marker_variant():
    body = "the final answer"
    for marker in (dl.RECOVERED_MARKER, dl.RECONNECTED_MARKER, dl.FLOOD_MARKER):
        assert dl.normalize_content(marker + body) == body
    assert dl.normalize_content("  the final answer  ") == body
    assert dl.normalize_content(None) == ""


# --------------------------------------------------------------- delivered stays delivered

def test_delivered_row_is_not_resurrected_by_a_marker_variant():
    """A delivered answer must not be turned back into 'pending' by a re-record: that is exactly
    how one answer gets sent twice."""
    _record("ob-done", content="the final answer")
    with dl._connect() as conn:
        conn.execute("UPDATE delivery_obligations SET state='delivered' WHERE obligation_id='ob-done'")
    _record("ob-done", content=dl.RECOVERED_MARKER + "the final answer")
    assert _row("ob-done")["state"] == "delivered", (
        "a delivered answer must not be resurrected into 'pending' and sent again"
    )


# --------------------------------------------------------------- reverse injection (must fire)

def test_a_row_owned_by_another_live_process_is_not_abandoned(monkeypatch):
    """Reverse injection: the fix must not reach into another live gateway's work.

    Only a row this process owns, or one whose owner is dead, may be judged. A row another
    LIVE process owns is that owner's business, budget spent or not.
    """
    _record("ob-other")
    _set_state("ob-other", "failed", dl.STALE_SESSION_ATTEMPTS, STALE_ERROR,
               owner_pid=424242, owner_started_at=12345.0)
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: True)

    dl.sweep_recoverable(deliverable_platforms={"weixin"})

    assert _row("ob-other")["state"] == "failed", (
        "a row another live process owns must be left alone, never judged by this boot"
    )


def test_pid_reuse_does_not_make_a_foreign_row_ours(monkeypatch):
    """Reverse injection: a dead process's row can carry OUR pid with a NULL start time.

    Treating that as 'ours' would let the boot sweep abandon a stranger's row. The start-time
    stamp is what makes ownership safe, so a NULL stamp is not our row.
    """
    _record("ob-pid")
    pid, _started = dl._owner_stamp()
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET state='failed', attempts=?, last_error=?, "
            "owner_pid=?, owner_started_at=NULL WHERE obligation_id=?",
            (dl.STALE_SESSION_ATTEMPTS, STALE_ERROR, pid, "ob-pid"))
    monkeypatch.setattr(dl, "_owner_alive", lambda p, started: True)

    dl.sweep_recoverable(deliverable_platforms={"weixin"})

    assert _row("ob-pid")["state"] == "failed", (
        "PID reuse must not make a foreign row ours: without a start-time stamp it is not ours"
    )
