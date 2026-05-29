#!/usr/bin/env bash
# Nightly Postgres backup for the Zeus trading DB.
#
# Why off-machine matters
# ───────────────────────
# Everything Zeus *learns* — model_versions, agent_journal, trades, 90 days
# of ohlcv_daily, the lessons table — lives in this DB. The Dockerfile +
# code are reproducible from git, but losing the DB means losing weeks of
# realized P&L history, the agent's prior reasoning, and any model that
# wasn't already promoted via artifact_path. A single Docker Desktop wipe
# or Windows reinstall destroys all of it.
#
# This script:
#   1. pg_dump the entire DB to a timestamped .dump file (custom format,
#      compressed, fastest for pg_restore).
#   2. Rotate locally — keep the last $LOCAL_RETENTION_DAYS dumps.
#   3. If BACKUP_REMOTE_CMD is set, pipe the dump to it (rclone, aws s3 cp,
#      restic, scp — anything that reads stdin or takes a path arg).
#
# Configure via env (zeus's .env file already loaded by docker-compose):
#   DB_PASSWORD                   — required, set in .env
#   BACKUP_DIR                    — local dump dir (default /backups)
#   LOCAL_RETENTION_DAYS          — keep N days locally (default 14)
#   BACKUP_REMOTE_CMD             — optional. Receives the dump path as $1.
#                                   Example: "aws s3 cp \$1 s3://zeus-backups/"
#
# Run from compose:  docker compose run --rm zeus-backup
# Or manually:       BACKUP_REMOTE_CMD="..." bash scripts/backup_db.sh
set -euo pipefail

: "${DB_PASSWORD:?DB_PASSWORD must be set (export from .env)}"
DB_HOST="${DB_HOST:-postgres}"
DB_USER="${DB_USER:-zeus}"
DB_NAME="${DB_NAME:-zeus_db}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
LOCAL_RETENTION_DAYS="${LOCAL_RETENTION_DAYS:-14}"

mkdir -p "$BACKUP_DIR"
timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
dump_path="${BACKUP_DIR}/zeus_db_${timestamp}.dump"

echo "[backup] starting pg_dump → ${dump_path}"
PGPASSWORD="$DB_PASSWORD" pg_dump \
    --host="$DB_HOST" \
    --username="$DB_USER" \
    --dbname="$DB_NAME" \
    --format=custom \
    --compress=6 \
    --no-owner \
    --no-privileges \
    --file="$dump_path"

dump_size=$(stat -c %s "$dump_path" 2>/dev/null || stat -f %z "$dump_path")
echo "[backup] dump complete: ${dump_size} bytes"

# Local rotation — keep last N days. find -mtime is portable enough for
# the Postgres alpine image (busybox find).
echo "[backup] rotating local dumps older than ${LOCAL_RETENTION_DAYS}d"
find "$BACKUP_DIR" -maxdepth 1 -name 'zeus_db_*.dump' \
    -mtime "+${LOCAL_RETENTION_DAYS}" -delete || true

# Off-machine upload — only attempted when BACKUP_REMOTE_CMD is set, so a
# minimum-config deployment still gets local backups without surfacing
# upload errors.
if [[ -n "${BACKUP_REMOTE_CMD:-}" ]]; then
    echo "[backup] off-machine upload via BACKUP_REMOTE_CMD"
    # Pass the dump path as $1 so the command can reference it however it
    # wants (aws s3 cp, rclone copyto, restic backup, scp, etc.).
    if bash -c "$BACKUP_REMOTE_CMD" _ "$dump_path"; then
        echo "[backup] off-machine upload complete"
    else
        # Local copy is preserved on remote-upload failure. Surface a
        # non-zero exit so docker-compose marks the run failed and the
        # scheduler sees it in `job_executions`.
        echo "[backup] off-machine upload FAILED — local dump retained at ${dump_path}" >&2
        exit 2
    fi
else
    echo "[backup] BACKUP_REMOTE_CMD not set — local-only backup. Set this to upload off-machine (see header)."
fi

echo "[backup] done"
