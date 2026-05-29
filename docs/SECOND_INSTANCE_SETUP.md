# Running a second Zeus instance on a second machine

This runbook gets you from "I copied the folder to an external drive" to
"two independent Zeus instances trading on two independent paper
accounts." The two systems share *nothing at runtime* — separate
Postgres, separate Redis, separate Alpaca account, separate Anthropic
spend budget, separate Telegram channel.

What the two instances **will** share (you copy them from the source):

- The code (`zeus/`, `scripts/`, `tests/`).
- The model artifacts (`artifacts/`) — joblib bundles are read-only at
  runtime and reusable. Optional; you can also start with no models and
  let the new instance train its own from scratch.
- The docs (`docs/`).

What the two instances **must not** share:

- `data/postgres/` — Postgres won't start with a `postmaster.pid` it
  didn't write, and even if it did you'd inherit broker positions from
  account #1 that account #2 doesn't actually hold.
- `data/redis/` — stale macro feature cache.
- `data/backups/` — these are pg_dumps of account #1's DB; useless on
  the second machine and confusing if they look like recoverable state.
- `.env` — must be edited to use account #2's Alpaca keys before boot.

The hard constraint that makes this safe: **a different
`ALPACA_API_KEY`**. As long as the two machines target different paper
accounts, every other piece of state is local to each machine and the
two systems cannot conflict.

---

## Step 1 — On the source machine, prepare the copy

You can copy the whole folder, but you'll save time and avoid
confusion if you exclude the runtime data directories before copying.

PowerShell on the source machine:

```powershell
# Either: copy first, clean later (see Step 4 on the destination).
# Or: pre-clean an intermediate directory before copying to the drive.

# Pre-clean approach (recommended). Copy the project somewhere on the
# external drive, *excluding* the per-instance state dirs and log/cache
# clutter.
$src = "C:\Users\guoje\Downloads\Zeus Trading Agent"
$dst = "E:\zeus-instance-2"   # path on your external drive
robocopy $src $dst /MIR /XD `
  "$src\data\postgres" `
  "$src\data\redis" `
  "$src\data\backups" `
  "$src\logs" `
  "$src\.pytest_cache" `
  /XF "*.pyc"
```

`/MIR` mirrors the tree, `/XD` excludes directories, `/XF` excludes
files. `__pycache__` directories will be copied but Python rebuilds
them automatically; they don't cause problems.

**Note**: `artifacts/` is included. The trained models inside are
reusable. If you want the second instance to learn from scratch
instead, also pass `"$src\artifacts"` to `/XD`.

## Step 2 — Get a second Alpaca paper account + API keys

1. Go to <https://app.alpaca.markets/signup> (or sign in if you already
   have a different account on a different email).
2. The default account is paper. Confirm by going to the dashboard and
   verifying "PAPER" is shown in the top-right.
3. In the dashboard, generate API keys. Save the **API Key ID** and the
   **Secret Key** somewhere safe — Alpaca only shows the secret once.

## Step 3 — On the destination machine, install prerequisites

You need Docker Desktop. On Windows that's downloadable from
<https://www.docker.com/products/docker-desktop/>. Verify it's running:

```powershell
docker version
docker compose version
```

Both should print version numbers. If `docker compose` doesn't work but
`docker-compose` does, you're on the old standalone version — fine,
substitute it in the commands below.

## Step 4 — Copy the project from the external drive

```powershell
# Pick a project root on the destination machine.
$dst = "C:\zeus"
mkdir $dst -ErrorAction SilentlyContinue
robocopy "E:\zeus-instance-2" $dst /MIR
```

If you skipped pre-cleaning in Step 1, run the cleanup helper now:

```powershell
cd C:\zeus
bash scripts/prepare_second_instance.sh
# or, if you don't have Git Bash:
Remove-Item -Recurse -Force data/postgres, data/redis, data/backups, logs -ErrorAction SilentlyContinue
```

The helper script just removes the four state directories. Both work.

## Step 5 — Edit `.env` with the new account's keys

Open `C:\zeus\.env` in your editor and change these fields:

| Field | Set to | Why |
|---|---|---|
| `ALPACA_API_KEY` | New paper account's key from Step 2 | **Hard requirement** — different from instance #1 |
| `ALPACA_SECRET_KEY` | New paper account's secret | Same |
| `DB_PASSWORD` | Pick a new strong password | Two databases, two passwords. Optional but recommended hygiene. |
| `DATABASE_URL` | Update the password portion to match the new `DB_PASSWORD` | Must match — Postgres reads `DB_PASSWORD`, the app reads `DATABASE_URL` |
| `TELEGRAM_BOT_TOKEN` | Leave blank, or use a different bot | If you leave the same token as instance #1, both will post to the same chat and you won't be able to tell which instance posted what |
| `TELEGRAM_CHAT_ID` | Different chat, or blank | Same reason |
| `ANTHROPIC_API_KEY` | Your existing key (fine to share) | LLM spend stacks per-key; both instances will count against the same daily budget. If you want isolated budgets, use a second key. |
| Everything else | Leave alone | `FRED_API_KEY`, `ENVIRONMENT=paper`, `LOG_LEVEL`, paths — instance-agnostic. |

## Step 6 — Bring up the database first, then run migrations

```powershell
cd C:\zeus
docker compose up -d zeus-postgres zeus-redis
# Wait until both report "healthy":
docker compose ps
```

Apply the alembic migrations to create every table from scratch:

```powershell
docker compose run --rm zeus-scheduler alembic upgrade head
```

You should see migrations 001 → 002 → 003 → 004 → 005 apply in order.
If the run command errors out with "image not found", first build:

```powershell
docker compose --profile trading build
```

## Step 7 — Seed the roadmap tracker

```powershell
docker compose run --rm zeus-scheduler python -m scripts.seed_roadmap
```

You should see `seeded 18/18 tasks`. (The Roadmap tab on the dashboard
needs this to show anything.)

## Step 8 — (Optional) Inherit the model bundles from instance #1

If you copied `artifacts/` from the source machine, you also need to
seed the `model_versions` table so the runtime knows about them. Two
options:

**Option 8a — Fresh start.** Skip this step. The scheduler will boot,
see no models in the registry, and fall back to `_NullPredictor` (zero
predictions, no trades). The Sunday 02:00 ET `nightly_retrain_job` will
train fresh models from your local OHLCV data — which means you need
~250 days of bars before the model is useful. The system will quietly
sit on $100k cash until then.

**Option 8b — Copy the model registry from instance #1.** Run a
targeted `pg_dump` on instance #1 limited to the `model_versions` table,
and restore it on instance #2:

```powershell
# On the source machine (instance #1):
docker exec zeus-postgres pg_dump -U zeus -d zeus_db `
  --table=model_versions --data-only --no-owner `
  --file=/tmp/model_versions.sql
docker cp zeus-postgres:/tmp/model_versions.sql `
  "C:\Users\guoje\Downloads\Zeus Trading Agent\data\backups\model_versions.sql"
```

Copy that `.sql` file across to the destination machine (USB stick or
similar), then on the destination machine:

```powershell
docker cp model_versions.sql zeus-postgres:/tmp/model_versions.sql
docker exec zeus-postgres psql -U zeus -d zeus_db -f /tmp/model_versions.sql
```

After this, both instances reference the same `artifact_path` strings,
and since you copied the actual joblib files via Step 1, the loader
will find them. The two instances will start from the same model state
but diverge from there.

## Step 9 — Bring up the trading profile

```powershell
docker compose --profile trading up -d
```

Wait ~30 seconds for the scheduler to finish booting (it pulls 250
days of bars on first run if the universe is empty — that can take a
few minutes the very first time).

## Step 10 — Verify both instances are independent

On instance #2:

```powershell
# Dashboard
start http://localhost:8000
# Heartbeat should land within a minute (now stamps on startup —
# no false-positive CRITICAL events on first boot)
curl http://localhost:8000/health
# Account info — should show the NEW paper account, with $100k fresh
curl http://localhost:8000/api/summary
```

Cross-check that the two instances are looking at *different broker
accounts*. The Alpaca dashboard at <https://app.alpaca.markets/> shows
the account whose API keys you're logged in with — confirm the API
key in instance #2's `.env` corresponds to the account you intended.

**If you suddenly see positions in instance #2 that match instance #1
exactly, stop everything.** That means both `.env` files have the same
Alpaca key. Fix `.env` on instance #2 and restart.

---

## Day-2 operations

Each instance is its own world:

- Logs land in each machine's local `./logs/`. Not shared.
- `pg_dump` backups land in each machine's local `./data/backups/`.
  Not shared.
- The Roadmap tab's `roadmap_tasks` rows live in each machine's local
  Postgres. If you mark `w1-attribution-table` complete on instance
  #1, instance #2 still shows it as not started until you mark it
  there too. This is intentional — the two instances are independent
  experiments.
- Model promotion is also local. If instance #1 trains a better model
  on Sunday and promotes it, instance #2 won't notice. To sync model
  state between them, repeat the Step 8b procedure.
- The Anthropic API key budget is *not* local — both instances count
  against the same daily spend cap unless you use different keys.

## Same-machine variant (if you ever want two instances on one host)

This runbook assumes two physical machines. If you instead want two
instances on the *same* machine, you need to:

1. Edit `docker-compose.yml` and change all five `container_name`
   values + the host port mappings (Postgres `5432→`, Redis `6379→`,
   monitor `8000→`) so the two stacks don't collide on the Docker
   socket.
2. Use a different project name when running compose: `docker compose
   -p zeus2 up -d`.
3. Mount different host volume paths (`./data/postgres` → `./data2/postgres`).

This is more work than the second-machine path and probably not worth
it unless you have a specific reason. The second-machine path keeps
everything cleanly isolated by the OS itself.

## Things that will go wrong if you skip steps

| Mistake | Symptom |
|---|---|
| Forgot to clear `data/postgres/` | Postgres container crash-loops with `LOG: lock file "postmaster.pid" already exists` |
| Same `ALPACA_API_KEY` in both .envs | Both instances see the same positions, both try to submit entries — exposure breaches, ghost trade rows, position drift events spam the risk log |
| Same `TELEGRAM_CHAT_ID` | Duplicate alerts, can't tell which instance fired which |
| Missed Step 8b but copied `artifacts/` | The artifacts directory has files but `model_versions` table is empty → predictor falls back to NullPredictor → no trades |
| Did Step 8b but `artifact_path` strings are absolute and point at machine #1's filesystem | Loader will fail with "model_artifact_missing" — quickest fix is to delete that `model_versions` row and re-train, or manually update the path |
| Didn't update `DATABASE_URL` after changing `DB_PASSWORD` | Postgres comes up healthy but the scheduler container can't connect — `psycopg2.OperationalError: password authentication failed` |
