"""Thin Anthropic client with prompt-caching + cost accounting.

This is the only module that talks to the Anthropic SDK. Agents don't call
the SDK directly — they go through `AnthropicClient.run()`, which:

  1. Adds the shared system prompt + tool spec as a prompt-cache block so
     repeated runs in a session only pay the small incremental-context cost.
  2. Loops on tool-use, calling registered tool handlers and feeding results
     back until the model issues a final text response (or hits the tool-use
     budget).
  3. Tracks input / output / cache-read / cache-creation tokens per call and
     converts to USD, enforcing a per-agent daily budget (settings-driven).

Intentionally does NOT depend on any specific agent — any agent can use it
by passing its own system prompt, tool spec, and message list.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from zeus.config.settings import get_settings

log = structlog.get_logger(__name__)


# ─── Pricing table ────────────────────────────────────────────────────────────
# Prices are per 1M tokens (input / output). Cache-read is 10% of input. Cache
# writes are 1.25x the input rate (Anthropic standard). Update when the
# published rates change.
_PRICING_USD_PER_MTOK: Dict[str, Dict[str, float]] = {
    "opus": {"input": 15.0, "output": 75.0},
    "sonnet": {"input": 3.0, "output": 15.0},
    "haiku": {"input": 0.80, "output": 4.0},
}


@dataclass
class LLMUsage:
    """Per-call token / cost breakdown."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMResponse:
    """Result of an AnthropicClient.run() call."""
    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)
    stop_reason: str = ""
    n_turns: int = 0


class BudgetExceededError(RuntimeError):
    """Raised when an agent's per-day LLM spend has been exhausted. Overseer
    halts the agent via RiskEngine.halt_agent(agent_id)."""


class AnthropicClient:
    """Wrapper around anthropic.Anthropic with tool-use loop + cost guards."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tier: str = "sonnet",
        max_turns: int = 10,
        daily_budget_usd: Optional[float] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ):
        self._tier = tier
        self._max_turns = max_turns
        self._settings = get_settings()
        self._api_key = api_key or self._settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._daily_budget = (
            daily_budget_usd
            if daily_budget_usd is not None
            else self._settings.llm_daily_budget_usd_per_agent
        )
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        # {agent_id -> {date_iso -> cumulative_cost_usd}}
        self._spend: Dict[str, Dict[str, float]] = {}
        self._client = None  # lazy — import anthropic only when first used
        # Params the active model has rejected with a "deprecated" 400.
        # Populated by _create_message self-heal; stripped from every
        # subsequent call so we don't pay the failed-call tax twice.
        self._deprecated_params: set[str] = set()

    # ─── Introspection ────────────────────────────────────────────────────────
    @property
    def model_id(self) -> str:
        return self._model_for_tier(self._tier)

    def _model_for_tier(self, tier: str) -> str:
        return {
            "opus": self._settings.llm_model_opus,
            "sonnet": self._settings.llm_model_sonnet,
            "haiku": self._settings.llm_model_haiku,
        }[tier]

    def agent_spend_today(self, agent_id: str) -> float:
        day = self._now().date().isoformat()
        return self._spend.get(agent_id, {}).get(day, 0.0)

    def remaining_budget_today(self, agent_id: str) -> float:
        return max(0.0, self._daily_budget - self.agent_spend_today(agent_id))

    # ─── Main entrypoint ──────────────────────────────────────────────────────
    def run(
        self,
        *,
        agent_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_handlers: Optional[Dict[str, Callable[..., Any]]] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        cache_system: bool = True,
    ) -> LLMResponse:
        """Run the model with a tool-use loop.

        - `system_prompt` is passed in a prompt-cache block if `cache_system` is True.
        - If the model returns tool-use content blocks, we resolve each via
          `tool_handlers[name](**input)`, append the tool_result, and keep looping
          until the model returns text or `max_turns` is hit.
        """
        if self.remaining_budget_today(agent_id) <= 0.0:
            raise BudgetExceededError(
                f"agent {agent_id!r} has exhausted its daily LLM budget "
                f"({self._daily_budget:.2f} USD)"
            )

        tools = tools or []
        tool_handlers = tool_handlers or {}
        usage = LLMUsage()
        conversation: List[Dict[str, Any]] = list(messages)

        system_blocks = self._format_system(system_prompt, cache_system)

        all_tool_calls: List[Dict[str, Any]] = []
        all_tool_results: List[Dict[str, Any]] = []
        final_text = ""
        stop_reason = ""
        turns = 0

        for turn in range(self._max_turns):
            turns = turn + 1
            call_kwargs: Dict[str, Any] = dict(
                system=system_blocks,
                messages=conversation,
                tools=tools,
                max_tokens=max_tokens,
            )
            # Newer Claude models (opus-4-7 onwards) rejected `temperature` with
            # a 400. Only forward it when the caller explicitly passed one.
            if temperature is not None:
                call_kwargs["temperature"] = temperature
            resp = self._create_message(**call_kwargs)
            turn_usage = self._extract_usage(resp)
            usage.input_tokens += turn_usage.input_tokens
            usage.output_tokens += turn_usage.output_tokens
            usage.cache_read_tokens += turn_usage.cache_read_tokens
            usage.cache_creation_tokens += turn_usage.cache_creation_tokens

            # Incremental cost accounting so we can short-circuit before making
            # the next turn if the budget is already blown.
            turn_cost = self._cost_usd(turn_usage)
            usage.cost_usd += turn_cost
            self._record_spend(agent_id, turn_usage, turn_cost)

            stop_reason = getattr(resp, "stop_reason", "")
            content = getattr(resp, "content", []) or []
            tool_use_blocks = [b for b in content if self._block_type(b) == "tool_use"]
            text_blocks = [b for b in content if self._block_type(b) == "text"]

            # Collect text from this turn (final-answer text is on the last turn)
            turn_text = "".join(self._block_text(b) for b in text_blocks)
            if turn_text:
                final_text = turn_text

            if not tool_use_blocks:
                break

            # Resolve each tool call and feed results back as a `user` message.
            assistant_msg = {"role": "assistant", "content": content}
            conversation.append(assistant_msg)

            tool_results_content: List[Dict[str, Any]] = []
            for block in tool_use_blocks:
                name = self._block_attr(block, "name")
                tool_input = self._block_attr(block, "input") or {}
                tool_id = self._block_attr(block, "id")
                all_tool_calls.append({"id": tool_id, "name": name, "input": tool_input})
                if name not in tool_handlers:
                    result_str = f"ERROR: no handler registered for tool {name!r}"
                    is_error = True
                else:
                    try:
                        raw = tool_handlers[name](**tool_input)
                        result_str = raw if isinstance(raw, str) else _json_dump(raw)
                        is_error = False
                    except Exception as e:
                        result_str = f"ERROR: {type(e).__name__}: {e}"
                        is_error = True
                all_tool_results.append({"id": tool_id, "name": name, "output": result_str})
                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_str,
                    "is_error": is_error,
                })
            conversation.append({"role": "user", "content": tool_results_content})

            if self.remaining_budget_today(agent_id) <= 0.0:
                log.warning(
                    "llm_budget_exhausted_midloop",
                    agent_id=agent_id, cost_so_far=usage.cost_usd,
                )
                break

        log.info(
            "llm_run_complete",
            agent_id=agent_id, tier=self._tier, n_turns=turns,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cache_read=usage.cache_read_tokens, cost_usd=round(usage.cost_usd, 4),
        )
        return LLMResponse(
            text=final_text,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            usage=usage,
            stop_reason=stop_reason,
            n_turns=turns,
        )

    # ─── Internals ────────────────────────────────────────────────────────────
    def _lazy_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise RuntimeError(
                    "anthropic SDK not installed. `pip install anthropic` first."
                ) from e
            if not self._api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set — agent-loop calls need this."
                )
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _create_message(self, **kwargs):
        """Retry with exponential backoff on transient failures."""
        client = self._lazy_client()
        # Proactively drop params this model has already rejected once.
        for p in self._deprecated_params:
            kwargs.pop(p, None)
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(4):
            try:
                return client.messages.create(model=self.model_id, **kwargs)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                # Self-heal on a deterministic 400 where the model has
                # deprecated a param we sent (e.g. `temperature` on opus-4-7).
                # Strip the offending key and retry immediately; remember it
                # so subsequent calls skip the failed round-trip entirely.
                if "deprecated" in msg and "temperature" in msg and "temperature" in kwargs:
                    self._deprecated_params.add("temperature")
                    kwargs.pop("temperature", None)
                    log.warning(
                        "llm_stripped_deprecated_param",
                        param="temperature", model=self.model_id,
                    )
                    continue
                # Only retry on rate-limit / overload / transient errors.
                if not _is_retryable(e) or attempt == 3:
                    raise
                log.warning("llm_retry", attempt=attempt, error=str(e), delay=delay)
                time.sleep(delay)
                delay *= 2
        if last_exc is None:  # pragma: no cover
            raise RuntimeError("_create_message exhausted with no exception captured")
        raise last_exc  # pragma: no cover

    @staticmethod
    def _format_system(system_prompt: str, cache: bool) -> List[Dict[str, Any]]:
        if not cache:
            return [{"type": "text", "text": system_prompt}]
        return [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]

    @staticmethod
    def _extract_usage(resp) -> LLMUsage:
        u = getattr(resp, "usage", None)
        if u is None:
            return LLMUsage()
        return LLMUsage(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        )

    def _cost_usd(self, usage: LLMUsage) -> float:
        rates = _PRICING_USD_PER_MTOK[self._tier]
        input_cost = usage.input_tokens * rates["input"] / 1_000_000.0
        output_cost = usage.output_tokens * rates["output"] / 1_000_000.0
        # Cache reads are 0.10x input rate; cache writes are 1.25x input rate.
        cache_read_cost = usage.cache_read_tokens * rates["input"] * 0.10 / 1_000_000.0
        cache_write_cost = usage.cache_creation_tokens * rates["input"] * 1.25 / 1_000_000.0
        return input_cost + output_cost + cache_read_cost + cache_write_cost

    def _record_spend(
        self, agent_id: str, turn_usage: LLMUsage, cost_usd: float
    ) -> None:
        day_dt = self._now().date()
        day_iso = day_dt.isoformat()
        self._spend.setdefault(agent_id, {})[day_iso] = (
            self._spend.get(agent_id, {}).get(day_iso, 0.0) + cost_usd
        )
        # Persist to agent_daily_spend so the overseer + dashboard can see it
        # across processes. Best-effort — a DB hiccup must not sink an LLM call.
        try:
            from zeus.data.storage.database import (
                AgentDailySpend,
                get_session_factory,
            )

            SessionLocal = get_session_factory()
            session = SessionLocal()
            try:
                row = session.get(AgentDailySpend, (agent_id, day_dt))
                if row is None:
                    row = AgentDailySpend(
                        agent_id=agent_id,
                        date=day_dt,
                        input_tokens=turn_usage.input_tokens,
                        output_tokens=turn_usage.output_tokens,
                        cache_read_tokens=turn_usage.cache_read_tokens,
                        cache_write_tokens=turn_usage.cache_creation_tokens,
                        calls=1,
                        cost_usd=cost_usd,
                    )
                    session.add(row)
                else:
                    row.input_tokens = (row.input_tokens or 0) + turn_usage.input_tokens
                    row.output_tokens = (row.output_tokens or 0) + turn_usage.output_tokens
                    row.cache_read_tokens = (
                        (row.cache_read_tokens or 0) + turn_usage.cache_read_tokens
                    )
                    row.cache_write_tokens = (
                        (row.cache_write_tokens or 0) + turn_usage.cache_creation_tokens
                    )
                    row.calls = (row.calls or 0) + 1
                    row.cost_usd = (row.cost_usd or Decimal(0)) + Decimal(str(cost_usd))
                session.commit()
            except Exception as e:
                session.rollback()
                log.warning(
                    "llm_spend_persist_failed", agent_id=agent_id, error=str(e)
                )
            finally:
                session.close()
        except Exception as e:
            log.warning(
                "llm_spend_persist_skipped", agent_id=agent_id, error=str(e)
            )

    # ─── Content-block accessors (SDK-shape tolerant) ────────────────────────
    @staticmethod
    def _block_type(b) -> str:
        return getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else "") or ""

    @staticmethod
    def _block_text(b) -> str:
        return getattr(b, "text", None) or (b.get("text", "") if isinstance(b, dict) else "") or ""

    @staticmethod
    def _block_attr(b, name: str):
        return getattr(b, name, None) if not isinstance(b, dict) else b.get(name)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"APIStatusError", "APIConnectionError", "APITimeoutError"}:
        return True
    status = getattr(exc, "status_code", None)
    if status in {429, 500, 502, 503, 504, 529}:
        return True
    return False


def _json_dump(x: Any) -> str:
    import json

    def _default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)

    return json.dumps(x, default=_default, ensure_ascii=False)
