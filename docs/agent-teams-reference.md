# Agent Teams — Master Reference Guide

> Internal reference for Claude when designing and orchestrating agent teams in this project.
> Source: https://code.claude.com/docs/en/agent-teams
> Requires Claude Code v2.1.32+. Experimental feature.

---

## 1. What Agent Teams Are

Agent teams coordinate **multiple independent Claude Code sessions** working together:

- **One Team Lead** — the main session; creates the team, spawns teammates, assigns work, synthesizes results.
- **Teammates** — separate Claude Code instances, each with its own context window.
- **Shared Task List** — coordinated work queue with dependencies and file-locked claiming.
- **Mailbox** — direct agent-to-agent messaging (1:1 and broadcast).

Unlike subagents, teammates **communicate directly with each other** and the user can **message any teammate directly** without going through the lead.

---

## 2. Agent Teams vs Subagents — Decision Matrix

| Dimension | Subagents | Agent Teams |
|---|---|---|
| Context | Own window; result returns to caller | Own window; fully independent |
| Communication | Report back to main agent only | Teammates talk to each other directly |
| Coordination | Main agent manages all work | Shared task list + self-coordination |
| Token cost | Lower (results summarized) | Higher (each teammate is a full instance) |
| Best for | Focused tasks where only the result matters | Complex work needing discussion/collaboration |

**Rule of thumb:**
- Need workers to **challenge each other, share findings, debate**? → Agent team.
- Need **quick focused workers that report back**? → Subagents.
- **Sequential task, same-file edits, heavy dependencies**? → Single session.

---

## 3. When to Use Agent Teams — Strongest Use Cases

1. **Research and review** — multiple angles investigated simultaneously, findings cross-challenged.
2. **New modules / features** — each teammate owns a separate piece, no file conflicts.
3. **Debugging with competing hypotheses** — adversarial investigation converges on root cause faster than sequential exploration (which suffers anchoring bias).
4. **Cross-layer coordination** — frontend / backend / tests each owned by a different teammate.

**Anti-patterns (do NOT use a team):**
- Sequential tasks with strict ordering.
- Work where multiple teammates would edit the same file.
- Routine single-threaded tasks.
- Tasks small enough that coordination overhead > parallelism benefit.

---

## 4. Enabling Agent Teams

Set the env var in `settings.json` (project local, user, or shell):

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

**Project-local path:** `.claude/settings.local.json`
**User path:** `~/.claude/settings.json`

Restart Claude Code after setting.

---

## 5. Starting a Team

Describe the task and team structure in natural language. Claude creates the team, spawns teammates, coordinates work.

**Good spawn prompt pattern** — independent roles, distinct lenses:

```text
I'm designing a CLI tool that helps developers track TODO comments across
their codebase. Create an agent team to explore this from different angles:
one teammate on UX, one on technical architecture, one playing devil's
advocate.
```

**Specify team size and model explicitly when needed:**

```text
Create a team with 4 teammates to refactor these modules in parallel.
Use Sonnet for each teammate.
```

**Name teammates you'll reference later** — tell the lead what to call each one so subsequent prompts can target them by name.

---

## 6. Display Modes

| Mode | Behavior | Requirements |
|---|---|---|
| **in-process** | All teammates in main terminal; Shift+Down to cycle | Any terminal |
| **split panes** | Each teammate in its own pane | tmux or iTerm2 + `it2` CLI |
| **auto** (default) | Split if already in tmux, else in-process | — |

**Set globally** in `~/.claude.json`:
```json
{ "teammateMode": "in-process" }
```

**Per-session override:**
```bash
claude --teammate-mode in-process
```

**In-process navigation:**
- `Shift+Down` — cycle through teammates (wraps to lead after last)
- `Enter` — enter a teammate's session view
- `Escape` — interrupt a teammate's current turn
- `Ctrl+T` — toggle task list

**Split-pane requirements:**
- tmux via system package manager (best on macOS; `tmux -CC` in iTerm2 recommended)
- OR iTerm2 + `it2` CLI + enable Python API in Settings → General → Magic

**NOT supported for split panes:** VS Code integrated terminal, Windows Terminal, Ghostty.

---

## 7. Architecture

```
┌─────────────┐
│  Team Lead  │  ← creates team, assigns tasks, synthesizes
└──────┬──────┘
       │ spawns + messages
       ▼
┌────────────────────────────────┐
│  Teammates (own context each)  │  ← message each other directly
└────────┬───────────────────────┘
         │
         ▼
   ┌──────────────┐      ┌──────────┐
   │ Task List    │◄────►│ Mailbox  │
   │ (shared)     │      │          │
   └──────────────┘      └──────────┘
```

**Storage (auto-generated, DO NOT hand-edit):**
- Team config: `~/.claude/teams/{team-name}/config.json`
- Task list: `~/.claude/tasks/{team-name}/`

The config `members` array contains each teammate's name, agent ID, and agent type. Teammates can read this to discover each other. Manual edits get overwritten on next state update.

**No project-level team config.** A `.claude/teams/teams.json` in your project is treated as an ordinary file, not configuration.

---

## 8. Task List Mechanics

- Three states: **pending**, **in progress**, **completed**.
- Tasks can **depend on other tasks**. Pending tasks with unresolved deps can't be claimed.
- **File locking** prevents race conditions when multiple teammates claim simultaneously.
- Dependents **auto-unblock** when the task they depend on completes.

**Assignment modes:**
- **Lead assigns** — tell the lead which task goes to which teammate.
- **Self-claim** — after finishing, a teammate picks up the next unassigned, unblocked task.

---

## 9. Messaging

| Type | Use | Cost |
|---|---|---|
| `message` | 1:1 to a specific teammate | Single recipient |
| `broadcast` | All teammates | **Scales with team size — use sparingly** |

- Messages delivered **automatically** — no polling required.
- Teammates **auto-notify the lead** when they go idle.
- Any teammate can message any other by name.

---

## 10. Reusing Subagent Definitions as Teammates

Spawn a teammate using a subagent type from any scope (project, user, plugin, CLI-defined):

```text
Spawn a teammate using the security-reviewer agent type to audit the auth module.
```

**What gets applied:**
- ✅ `tools` allowlist
- ✅ `model`
- ✅ Body appended to system prompt (NOT replacing it)

**What gets IGNORED when used as a teammate:**
- ❌ `skills` frontmatter
- ❌ `mcpServers` frontmatter

Teammates load skills and MCP servers from project/user settings, same as a regular session.

**Team coordination tools** (`SendMessage`, task management) are **always available** even when `tools` restricts everything else.

---

## 11. Plan Approval Gating

For risky work, require a teammate to plan before acting. Teammate works in **read-only plan mode** until the lead approves.

```text
Spawn an architect teammate to refactor the authentication module.
Require plan approval before they make any changes.
```

**Flow:**
1. Teammate produces a plan, sends approval request to lead.
2. Lead reviews, approves OR rejects with feedback.
3. On rejection, teammate revises in plan mode and resubmits.
4. On approval, teammate exits plan mode and implements.

**Steering the lead's judgment** — include criteria in your prompt:
- "only approve plans that include test coverage"
- "reject plans that modify the database schema"

---

## 12. Context Loaded by Each Teammate

Each teammate starts fresh with:
- ✅ `CLAUDE.md` files (from working directory)
- ✅ MCP servers (project + user settings)
- ✅ Skills (project + user settings)
- ✅ The spawn prompt from the lead

**NOT loaded:**
- ❌ The lead's conversation history

**Implication:** the spawn prompt must contain all task-specific context the teammate needs.

---

## 13. Permissions

- Teammates **inherit the lead's permission settings at spawn time**.
- If lead runs with `--dangerously-skip-permissions`, all teammates do too.
- You **can** change individual teammate modes after spawning.
- You **cannot** set per-teammate modes at spawn time.
- Teammate permission prompts **bubble up to the lead** → pre-approve common ops before spawning to reduce friction.

---

## 14. Hooks for Quality Gates

Use hooks in `settings.json` to enforce rules:

| Hook | Fires when | Exit code 2 effect |
|---|---|---|
| `TeammateIdle` | Teammate about to go idle | Send feedback, keep working |
| `TaskCreated` | Task being created | Prevent creation + feedback |
| `TaskCompleted` | Task being marked complete | Prevent completion + feedback |

---

## 15. Shutdown & Cleanup

**Shut down a single teammate:**
```text
Ask the researcher teammate to shut down
```
Teammate can approve (exits gracefully) or reject with explanation.

**Clean up the whole team:**
```text
Clean up the team
```

**CRITICAL:** Always use the **lead** to clean up. Cleanup fails if any teammate is still active — shut them down first. Teammates themselves should NOT run cleanup (their team context may not resolve correctly).

---

## 16. Best Practices

### 16.1 Team sizing
- **Default: 3–5 teammates.** Balances parallelism with coordination overhead.
- Token cost scales linearly with teammate count.
- Three focused teammates > five scattered ones.
- Scale up only when work *genuinely* benefits from more parallelism.

### 16.2 Task sizing
- **Target: 5–6 tasks per teammate** — keeps everyone productive, lets the lead reassign if someone stalls.
- Too small → coordination overhead exceeds benefit.
- Too large → long runs without check-ins, wasted effort risk.
- Just right → self-contained unit with a clear deliverable (a function, a test file, a review).

### 16.3 Spawn prompts
Load each teammate with:
- The concrete task boundary
- Relevant file paths / modules
- Domain context not in CLAUDE.md
- Output format expected
- Severity / prioritization criteria if reporting findings

Example:
```text
Spawn a security reviewer teammate with the prompt: "Review the authentication
module at src/auth/ for security vulnerabilities. Focus on token handling,
session management, and input validation. The app uses JWT tokens stored in
httpOnly cookies. Report any issues with severity ratings."
```

### 16.4 Avoid file conflicts
Two teammates editing the same file → overwrites. **Partition by file ownership.**

### 16.5 Keep lead delegating, not doing
If the lead starts implementing instead of waiting:
```text
Wait for your teammates to complete their tasks before proceeding
```

### 16.6 Monitor and steer
- Check in on progress.
- Redirect approaches that aren't working.
- Synthesize findings as they arrive.
- Unattended teams = wasted-effort risk.

### 16.7 Start easy
First-time use → pick tasks with clear boundaries and no code writing:
- PR review
- Library research
- Bug investigation
These show parallel-exploration value without parallel-implementation coordination pain.

---

## 17. Proven Prompt Patterns

### 17.1 Parallel code review (distinct lenses)
```text
Create an agent team to review PR #142. Spawn three reviewers:
- One focused on security implications
- One checking performance impact
- One validating test coverage
Have them each review and report findings.
```

### 17.2 Adversarial debugging (competing hypotheses)
```text
Users report the app exits after one message instead of staying connected.
Spawn 5 agent teammates to investigate different hypotheses. Have them talk
to each other to try to disprove each other's theories, like a scientific
debate. Update the findings doc with whatever consensus emerges.
```
**Why this works:** sequential investigation anchors on the first plausible theory. Adversarial teammates actively trying to disprove each other surface the actual root cause.

### 17.3 Parallel refactor with plan gating
```text
Create a team with N teammates to refactor [modules] in parallel. Each
teammate owns a separate module. Require plan approval before any changes.
Use Sonnet. Only approve plans that include test coverage.
```

### 17.4 Research with synthesis
```text
Create a team of 3 teammates to evaluate libraries X, Y, Z for [use case].
Each takes one library. Compare on API ergonomics, performance, maintenance
status, and license. Lead synthesizes a recommendation matrix at the end.
```

---

## 18. Troubleshooting

| Symptom | Diagnosis / Fix |
|---|---|
| Teammates not appearing | In-process: press Shift+Down. Task may be too simple → Claude didn't spawn any. Split-pane: check `which tmux` or verify `it2` + Python API. |
| Too many permission prompts | Pre-approve common ops in permission settings before spawning. |
| Teammate stopped on error | Shift+Down to view output → give direct instructions OR spawn a replacement. |
| Lead shuts down before work done | Tell it to keep going. If doing work instead of delegating, tell it to wait for teammates. |
| Task stuck (not marked complete) | Verify work is actually done. Update status manually OR tell lead to nudge teammate. |
| Orphaned tmux session | `tmux ls` then `tmux kill-session -t <name>` |

---

## 19. Known Limitations (as of doc date)

- **No session resumption with in-process teammates** — `/resume` and `/rewind` do NOT restore them. Lead may try to message non-existent teammates → spawn new ones.
- **Task status can lag** — teammates sometimes fail to mark tasks complete, blocking dependents.
- **Shutdown is slow** — teammates finish current request/tool call before exiting.
- **One team per session** — clean up before starting a new one.
- **No nested teams** — teammates cannot spawn their own teams.
- **Lead is fixed** — cannot promote a teammate or transfer leadership.
- **Permissions set at spawn** — no per-teammate modes at spawn time.
- **Split panes** unsupported in VS Code terminal, Windows Terminal, Ghostty.

---

## 20. Quick Decision Tree (for future me)

```
Task arrives
│
├─ Trivial / sequential / single-file? ──────► Single session
│
├─ Multiple focused lookups, results only? ──► Subagents
│
├─ Needs parallel exploration + cross-talk? ─► Agent team
│    │
│    ├─ Research / review?        → 3 teammates, distinct lenses
│    ├─ Debugging, unclear cause? → 5 teammates, adversarial
│    ├─ Multi-module refactor?    → 1 teammate per module, plan gating
│    └─ Cross-layer feature?      → 1 teammate per layer, file-partitioned
│
└─ Always: spawn with full context, monitor, synthesize, clean up via lead.
```

---

## 21. Checklist Before Spawning a Team

- [ ] Task genuinely benefits from parallelism (not sequential)
- [ ] Can partition work so teammates don't edit the same files
- [ ] `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set
- [ ] Team size chosen (default 3–5)
- [ ] Each teammate has a clear role / lens / file boundary
- [ ] Spawn prompts include task-specific context (not inherited from lead)
- [ ] Named teammates if you'll reference them later
- [ ] Plan-approval gating decided for risky work
- [ ] Permission mode appropriate (remember: inherited at spawn)
- [ ] Display mode appropriate (in-process vs split-pane)
- [ ] Cleanup plan: finish → shut down teammates → lead cleans up
