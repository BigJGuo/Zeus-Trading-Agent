#!/usr/bin/env bash
# Wipe per-instance runtime state so a freshly-copied Zeus folder boots
# cleanly on a second machine against its OWN paper account, with NO
# inherited state from the source machine.
#
# What this removes
# ─────────────────
#   data/postgres/     — source machine's Postgres data dir; Postgres
#                        refuses to start with a foreign postmaster.pid
#                        and even if it didn't, you'd inherit positions
#                        that belong to a different broker account
#   data/redis/        — stale macro feature cache, no value
#   data/backups/      — pg_dumps of the source machine's DB
#   logs/              — application logs from the source machine
#   .pytest_cache/     — pytest cache, irrelevant on fresh machine
#   __pycache__/       — compiled-bytecode cache, Python rebuilds
#
# What this PRESERVES
# ───────────────────
#   .env               — you MUST manually edit this (Alpaca keys etc.)
#                        before booting; see docs/SECOND_INSTANCE_SETUP.md
#   artifacts/         — trained model bundles; reusable across instances
#                        if you also seed model_versions (Step 8b)
#   zeus/, scripts/, tests/, docs/, alembic/   — code, no instance state
#
# Run from the project root:
#   bash scripts/prepare_second_instance.sh
#
# After this finishes, edit .env, then follow docs/SECOND_INSTANCE_SETUP.md
# starting at Step 5.
set -euo pipefail

if [[ ! -d "zeus" ]] || [[ ! -f "docker-compose.yml" ]]; then
    echo "[error] doesn't look like the Zeus project root — expected"
    echo "        ./zeus/ and ./docker-compose.yml to exist. Run this"
    echo "        from the directory where docker-compose.yml lives."
    exit 2
fi

# Re-confirm before deleting — `data/postgres` in particular is the
# kind of directory you do NOT want to nuke on the wrong machine.
echo "About to remove the following directories from $(pwd):"
echo "  - data/postgres/"
echo "  - data/redis/"
echo "  - data/backups/"
echo "  - logs/"
echo "  - .pytest_cache/"
echo
echo "This is destructive. Only run this on the *destination* machine"
echo "where you want a fresh-state Zeus instance."
read -rp "Proceed? Type 'yes' to confirm: " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "aborted"
    exit 1
fi

removed=()
for d in data/postgres data/redis data/backups logs .pytest_cache; do
    if [[ -e "$d" ]]; then
        rm -rf "$d"
        removed+=("$d")
    fi
done

# Sweep stray __pycache__ — these aren't dangerous, just clutter.
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

echo
echo "[done] removed: ${removed[*]:-<nothing — already clean>}"
echo
echo "Next:"
echo "  1. Edit .env — change ALPACA_API_KEY, ALPACA_SECRET_KEY,"
echo "     DB_PASSWORD (update DATABASE_URL to match), TELEGRAM_*"
echo "  2. docker compose up -d zeus-postgres zeus-redis"
echo "  3. docker compose run --rm zeus-scheduler alembic upgrade head"
echo "  4. docker compose run --rm zeus-scheduler python -m scripts.seed_roadmap"
echo "  5. docker compose --profile trading up -d"
echo
echo "Full runbook: docs/SECOND_INSTANCE_SETUP.md"
