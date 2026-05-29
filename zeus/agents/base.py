"""AgentLoop — thin runtime tying an agent's journal + LLM + tools together.

Every role-specific agent (day_trader, swing_research, overseer, ...) is
built on top of one `AgentLoop` instance. The loop's responsibilities:

  1. Pull recent journal entries as the agent's memory context.
  2. Call AnthropicClient.run(...) with the role's system prompt, tools,
     the memory snippet, and caller-provided input (e.g. model predictions
     for traders, universe for research).
  3. Persist the LLM output into the journal with the appropriate `kind`.
  4. For trader agents, hand collected `propose_trade` proposals off to
     the StrategyManager.

The loop is intentionally stateless — each `.run()` is self-contained so
APScheduler / CronCreate can fire it at the role's cadence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import structlog

from zeus.agents.journal import (
    AgentJournal,
    JournalEntry,
    KNOWN_AGENTS,
    PAIRED_RESEARCH,
    query_journal,
)
from zeus.llm.client import AnthropicClient, BudgetExceededError, LLMResponse
from zeus.llm.prompts import load_prompt
from zeus.llm.tools import ToolBundle, build_tool_bundle

log = structlog.get_logger(__name__)


@dataclass
class AgentRunResult:
    agent_id: str
    role: str
    text: str
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    journal_ids_written: List[int] = field(default_factory=list)
    cost_usd: float = 0.0
    n_turns: int = 0
    llm_response: Optional[LLMResponse] = None


# Role inferred from agent_id. Trader ids → 'trader', *_research → 'research', overseer.
def _role_for(agent_id: str) -> str:
    if agent_id == "overseer":
        return "overseer"
    if agent_id.endswith("_research"):
        return "research"
    return "trader"


# Which journal agents a given agent is allowed to read.
def _allowed_journal_agents(agent_id: str) -> List[str]:
    # Every live agent can also read the curated research library.
    library = "research_library"
    if agent_id == "overseer":
        return list(KNOWN_AGENTS)  # already includes research_library
    if agent_id.endswith("_research"):
        return [agent_id, library]
    # trader: self + paired research + library
    paired = PAIRED_RESEARCH.get(agent_id)
    return [agent_id, paired, library] if paired else [agent_id, library]


class AgentLoop:
    """One reusable runtime per agent."""

    def __init__(
        self,
        agent_id: str,
        *,
        llm: AnthropicClient,
        prompt_role: Optional[str] = None,      # override the filename if it isn't the agent_id
        broker: Optional[Any] = None,
        regime_fn: Optional[Callable[[], str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        memory_window_hours: int = 24,
        memory_limit: int = 20,
    ):
        if agent_id not in KNOWN_AGENTS:
            raise ValueError(f"unknown agent_id {agent_id!r}")
        self._agent_id = agent_id
        self._role = _role_for(agent_id)
        self._llm = llm
        # Trader agent IDs ("day", "swing", "long_term") don't match a
        # prompt file directly — the convention is `{agent_id}_trader.md`.
        # Resolve that here so callers can instantiate `AgentLoop("day")`
        # without having to know the file-naming convention.
        from zeus.agents.journal import TRADER_AGENTS
        if prompt_role is None and agent_id in TRADER_AGENTS:
            prompt_role = f"{agent_id}_trader"
        self._prompt_role = prompt_role or agent_id
        self._broker = broker
        self._regime_fn = regime_fn
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._memory_window_hours = memory_window_hours
        self._memory_limit = memory_limit
        self._journal = AgentJournal(agent_id=agent_id)
        self._system_prompt = load_prompt(self._prompt_role)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def role(self) -> str:
        return self._role

    @property
    def journal(self) -> AgentJournal:
        return self._journal

    # ─── Main run ─────────────────────────────────────────────────────────────
    def run(
        self,
        *,
        user_message: str,
        extra_context: Optional[Dict[str, Any]] = None,
        persist_output_as: Optional[str] = None,       # 'brief' | 'memo' | 'decision' | ...
        persist_title: Optional[str] = None,
        persist_symbol: Optional[str] = None,
    ) -> AgentRunResult:
        """Execute a single LLM loop for this agent.

        - `user_message` is the caller-supplied prompt (e.g. "Build tonight's
          swing briefs from these candidates: ..." + model predictions).
        - `extra_context` is a dict merged into the user message as JSON; use
          it for structured payloads (model preds, position list, etc.).
        - If `persist_output_as` is given, the final LLM text is recorded as
          a journal entry of that kind after the run completes.
        """
        bundle = self._build_tool_bundle()
        memory = self._build_memory_snippet()

        messages = self._build_messages(user_message, extra_context, memory)

        try:
            llm_resp = self._llm.run(
                agent_id=self._agent_id,
                system_prompt=self._system_prompt,
                messages=messages,
                tools=bundle.specs,
                tool_handlers=bundle.handlers,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except BudgetExceededError as e:
            log.error("agent_run_budget_exceeded", agent_id=self._agent_id, error=str(e))
            self._journal.record_alert(
                title="llm_budget_exhausted",
                body=str(e),
            )
            raise

        journal_ids: List[int] = []
        if persist_output_as and llm_resp.text.strip():
            jid = self._persist_output(
                persist_output_as, llm_resp.text, persist_title, persist_symbol,
            )
            if jid is not None:
                journal_ids.append(jid)

        # Trader proposals each become a `trade_rationale` linked later by the
        # orchestrator once StrategyManager assigns trade_ids — at this stage
        # we only have the proposal payload.
        for prop in bundle.proposals:
            jid = self._journal.record_rationale(
                symbol=prop.get("symbol", "?"),
                title=f"{prop.get('action', '?')} {prop.get('symbol', '?')}",
                body=prop.get("rationale", ""),
                structured=prop,
                confidence=prop.get("confidence") if prop.get("confidence") else None,
            )
            journal_ids.append(jid)

        return AgentRunResult(
            agent_id=self._agent_id,
            role=self._role,
            text=llm_resp.text,
            proposals=list(bundle.proposals),
            journal_ids_written=journal_ids,
            cost_usd=llm_resp.usage.cost_usd,
            n_turns=llm_resp.n_turns,
            llm_response=llm_resp,
        )

    # ─── Internals ────────────────────────────────────────────────────────────
    def _build_tool_bundle(self) -> ToolBundle:
        return build_tool_bundle(
            agent_id=self._agent_id,
            role=self._role,
            allowed_journal_agents=_allowed_journal_agents(self._agent_id),
            broker=self._broker,
            regime_fn=self._regime_fn,
        )

    def _build_memory_snippet(self) -> str:
        """Recent journal entries from this agent + (if trader) paired research,
        rendered into a compact text block the LLM reads as part of the user msg.
        """
        since = datetime.now(timezone.utc) - timedelta(hours=self._memory_window_hours)
        blocks: List[str] = []

        own = query_journal(
            agent_id=self._agent_id, since=since, limit=self._memory_limit,
        )
        if own:
            blocks.append(
                "## Recent entries (yours)\n"
                + "\n".join(self._fmt_entry(e) for e in own)
            )

        # Traders also get their paired research's recent briefs/memos.
        if self._role == "trader":
            paired = PAIRED_RESEARCH.get(self._agent_id)
            if paired:
                paired_entries = query_journal(
                    agent_id=paired, since=since, limit=self._memory_limit,
                )
                if paired_entries:
                    blocks.append(
                        f"## Recent entries ({paired})\n"
                        + "\n".join(self._fmt_entry(e) for e in paired_entries)
                    )
        return "\n\n".join(blocks)

    @staticmethod
    def _fmt_entry(e: JournalEntry) -> str:
        sym = f" [{e.symbol}]" if e.symbol else ""
        body_preview = e.body[:300] + ("…" if len(e.body) > 300 else "")
        return (
            f"- [{e.ts.isoformat()}] ({e.kind}){sym} id={e.id} "
            f"title={e.title!r}\n  {body_preview}"
        )

    @staticmethod
    def _build_messages(
        user_message: str,
        extra_context: Optional[Dict[str, Any]],
        memory: str,
    ) -> List[Dict[str, Any]]:
        import json

        parts: List[str] = []
        if memory:
            parts.append(memory)
        if extra_context:
            parts.append("## Input context\n```json\n" + json.dumps(
                extra_context, default=str, indent=2,
            ) + "\n```")
        parts.append("## Task\n" + user_message)
        return [{"role": "user", "content": "\n\n".join(parts)}]

    def _persist_output(
        self,
        kind: str,
        body: str,
        title: Optional[str],
        symbol: Optional[str],
    ) -> Optional[int]:
        title = title or f"{self._agent_id} {kind} {datetime.now(timezone.utc).isoformat(timespec='minutes')}"
        body = body.strip()
        try:
            if kind == "brief":
                return self._journal.record_brief(
                    symbol=symbol or "MARKET", title=title, body=body,
                )
            if kind == "memo":
                return self._journal.record_memo(
                    title=title, body=body, symbol=symbol,
                )
            if kind == "review":
                return self._journal.record_review(
                    symbol=symbol or "MARKET", title=title, body=body,
                )
            if kind == "decision":
                return self._journal.record_decision(title=title, body=body)
            if kind == "alert":
                return self._journal.record_alert(
                    title=title, body=body, symbol=symbol,
                )
            if kind == "lesson":
                return self._journal.record_lesson(title=title, body=body)
            if kind == "postmortem":
                return self._journal.record_postmortem(
                    title=title, body=body, symbol=symbol,
                )
            raise ValueError(f"unsupported persist_output_as kind {kind!r}")
        except Exception as e:
            log.error("persist_output_failed", kind=kind, error=str(e))
            return None
