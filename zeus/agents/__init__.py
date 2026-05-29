"""Agent layer for the 7-agent trading system.

Contains:
  - journal.py    — AgentJournal wrapper around the `agent_journal` table
  - metrics.py    — per-agent rolling Sharpe/hit/expectancy/R-multi
  - base.py       — AgentLoop (thin LLM + tools + journal runtime)
  - <role>.py     — one module per agent (day_trader, swing_research, overseer, ...)
"""
