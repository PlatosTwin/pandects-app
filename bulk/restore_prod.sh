#!/usr/bin/env bash
# Restore production MariaDB (pandects-db) from the promoted R2 logical backup.
#
# Scales the DB machine up for the load, runs restore_from_r2.py on a temporary
# pandects-bulk machine, verifies the result, and scales the DB machine back to
# its original size on exit. If the restore is still running when this script
# is interrupted or times out, the restore machine and the scaled-up DB are
# left alone (killing myloader mid-load would leave production partial).
#
# Usage:
#   MARIADB_PASSWORD=... bash bulk/restore_prod.sh [--keep-scaled]
#
# MARIADB_PASSWORD is the production `panda` password (prompted for if unset
# and stdin is a terminal). Overridable env: DB_APP, BULK_APP, RESTORE_VM_SIZE,
# RESTORE_VM_MEMORY, BULK_VM_SIZE, MYLOADER_THREADS, RESTORE_TIMEOUT_SECS.
set -euo pipefail

DB_APP=${DB_APP:-pandects-db}
BULK_APP=${BULK_APP:-pandects-bulk}
RESTORE_VM_SIZE=${RESTORE_VM_SIZE:-performance-2x}
RESTORE_VM_MEMORY=${RESTORE_VM_MEMORY:-4096}
BULK_VM_SIZE=${BULK_VM_SIZE:-performance-2x}
MYLOADER_THREADS=${MYLOADER_THREADS:-4}
RESTORE_TIMEOUT_SECS=${RESTORE_TIMEOUT_SECS:-10800}
DB_READY_TIMEOUT_SECS=${DB_READY_TIMEOUT_SECS:-600}
POLL_SECS=15

KEEP_SCALED=0
for arg in "$@"; do
  case "$arg" in
    --keep-scaled) KEEP_SCALED=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

command -v fly >/dev/null 2>&1 || die "fly is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
db_exec() {
  fly machine exec "$DB_MACHINE" --app "$DB_APP" "$1" 2>&1 || true
}

wait_for_db() {
  local waited=0
  while (( waited < DB_READY_TIMEOUT_SECS )); do
    if db_exec 'sh -c "mariadb-admin ping >/dev/null 2>&1 && echo DB_READY"' | grep -q DB_READY; then
      return 0
    fi
    sleep "$POLL_SECS"
    waited=$((waited + POLL_SECS))
  done
  return 1
}

# Runs one SQL statement as `panda` against pdx on the DB machine. The SQL may
# contain single quotes but not double quotes.
db_sql() {
  db_exec "sh -c \"mariadb -upanda -p\\\"\\\$MARIADB_PASSWORD\\\" -N -e \\\"$1\\\" pdx\""
}

# Prints "<id> <size-name> <memory_mb>" for the single DB machine.
db_machine_info() {
  fly machine list --app "$DB_APP" --json | python3 -c '
import json, sys
machines = json.load(sys.stdin)
if len(machines) != 1:
    sys.exit(f"expected exactly one machine in the DB app, found {len(machines)}")
m = machines[0]
guest = m["config"]["guest"]
kind = "shared-cpu" if guest["cpu_kind"] == "shared" else guest["cpu_kind"]
print(m["id"], "{}-{}x".format(kind, guest["cpus"]), guest["memory_mb"])
'
}

# Prints "<state> <exit_code|->" for the restore machine; empty on API failure.
restore_machine_status() {
  fly machine list --app "$BULK_APP" --json 2>/dev/null | python3 -c '
import json, sys
target = sys.argv[1]
try:
    machines = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
for m in machines:
    if m["id"] != target:
        continue
    code = "-"
    for ev in m.get("events") or []:
        if ev.get("type") == "exit":
            code = str(((ev.get("request") or {}).get("exit_event") or {}).get("exit_code", "-"))
            break
    print(m["state"], code)
    break
else:
    print("missing -")
' "$RESTORE_MACHINE" || true
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
fly auth whoami >/dev/null 2>&1 || die "not logged in to fly (run: fly auth login)"

if [[ -z "${MARIADB_PASSWORD:-}" ]]; then
  [[ -t 0 ]] || die "MARIADB_PASSWORD is not set"
  read -r -s -p "Production MariaDB password for user panda: " MARIADB_PASSWORD
  echo
  [[ -n "$MARIADB_PASSWORD" ]] || die "empty password"
fi

DB_INFO=$(db_machine_info)
read -r DB_MACHINE ORIG_VM_SIZE ORIG_VM_MEMORY <<< "$DB_INFO"
[[ -n "${ORIG_VM_MEMORY:-}" ]] || die "could not read the DB machine size"
log "DB machine $DB_MACHINE is currently $ORIG_VM_SIZE / ${ORIG_VM_MEMORY}MB"

BULK_IMAGE=$(fly releases --app "$BULK_APP" --json | python3 -c 'import json, sys; print(json.load(sys.stdin)[0]["ImageRef"])')
[[ -n "$BULK_IMAGE" ]] || die "could not resolve the latest $BULK_APP image"
log "Restore image: $BULK_IMAGE"
echo "    (deploy a fresh one first if the restore script changed: cd bulk && fly deploy --app $BULK_APP)"

# ---------------------------------------------------------------------------
# Cleanup: runs on every exit path
# ---------------------------------------------------------------------------
RESTORE_MACHINE=""
RESTORE_RUNNING=0
LOGS_PID=""
SCALED=0
RESTORE_OK=0

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$LOGS_PID" ]]; then kill "$LOGS_PID" 2>/dev/null; fi

  if [[ "$RESTORE_RUNNING" == 1 ]]; then
    log "⚠️  Leaving restore machine $RESTORE_MACHINE running and $DB_APP scaled up."
    echo "    Watch it:      fly logs --app $BULK_APP --machine $RESTORE_MACHINE"
    echo "    When it stops: fly machine destroy $RESTORE_MACHINE --app $BULK_APP --force"
    echo "    Then:          fly scale vm $ORIG_VM_SIZE --vm-memory $ORIG_VM_MEMORY --app $DB_APP"
    exit "$rc"
  fi

  if [[ -n "$RESTORE_MACHINE" ]]; then
    log "Destroying restore machine $RESTORE_MACHINE"
    fly machine destroy "$RESTORE_MACHINE" --app "$BULK_APP" --force >/dev/null 2>&1
  fi
  if [[ "$SCALED" == 1 ]]; then
    if [[ "$KEEP_SCALED" == 1 ]]; then
      log "Leaving $DB_APP at $RESTORE_VM_SIZE / ${RESTORE_VM_MEMORY}MB (--keep-scaled)."
      echo "    Scale back with: fly scale vm $ORIG_VM_SIZE --vm-memory $ORIG_VM_MEMORY --app $DB_APP"
    else
      log "Scaling $DB_APP back to $ORIG_VM_SIZE / ${ORIG_VM_MEMORY}MB"
      fly scale vm "$ORIG_VM_SIZE" --vm-memory "$ORIG_VM_MEMORY" --app "$DB_APP" \
        || echo "WARNING: scale-down failed; run: fly scale vm $ORIG_VM_SIZE --vm-memory $ORIG_VM_MEMORY --app $DB_APP" >&2
      wait_for_db || echo "WARNING: DB not ready after scale-down; check: fly logs --app $DB_APP" >&2
    fi
    fly scale show --app "$DB_APP"
  fi
  if [[ "$RESTORE_OK" == 1 && $rc -eq 0 ]]; then
    log "✅ Production restore complete."
  else
    log "❌ Restore did not complete (exit $rc). Production may be empty or partial — fix and rerun."
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------------------
# 1. Scale the DB machine up (downtime starts here)
# ---------------------------------------------------------------------------
log "Scaling $DB_APP to $RESTORE_VM_SIZE / ${RESTORE_VM_MEMORY}MB"
SCALED=1
fly scale vm "$RESTORE_VM_SIZE" --vm-memory "$RESTORE_VM_MEMORY" --app "$DB_APP"
log "Waiting for MariaDB to come back"
wait_for_db || die "MariaDB did not become ready within ${DB_READY_TIMEOUT_SECS}s"

# ---------------------------------------------------------------------------
# 2. Run the restore on a temporary bulk machine
# ---------------------------------------------------------------------------
RESTORE_NAME="restore-$(date +%Y%m%d-%H%M%S)"
log "Launching restore machine $RESTORE_NAME ($BULK_VM_SIZE, MYLOADER_THREADS=$MYLOADER_THREADS)"
fly machine run "$BULK_IMAGE" \
  --app "$BULK_APP" \
  --name "$RESTORE_NAME" \
  --detach \
  --restart no \
  --vm-size "$BULK_VM_SIZE" \
  --rootfs-size 40gb \
  --env MARIADB_HOST="${DB_APP}.internal" \
  --env MARIADB_PORT=3306 \
  --env MARIADB_USER=panda \
  --env MARIADB_PASSWORD="$MARIADB_PASSWORD" \
  --env MARIADB_DATABASE=pdx \
  --env MYLOADER_THREADS="$MYLOADER_THREADS" \
  python3 /app/restore_from_r2.py

RESTORE_MACHINE=$(fly machine list --app "$BULK_APP" --json | python3 -c '
import json, sys
name = sys.argv[1]
ids = [m["id"] for m in json.load(sys.stdin) if m.get("name") == name]
print(ids[0] if ids else "")
' "$RESTORE_NAME")
[[ -n "$RESTORE_MACHINE" ]] || die "restore machine $RESTORE_NAME was not created"
RESTORE_RUNNING=1
log "Restore machine id: $RESTORE_MACHINE — streaming its logs"

fly logs --app "$BULK_APP" --machine "$RESTORE_MACHINE" &
LOGS_PID=$!

waited=0
STATE=""
EXIT_CODE="-"
while (( waited < RESTORE_TIMEOUT_SECS )); do
  STATUS=$(restore_machine_status)
  STATE=${STATUS%% *}
  EXIT_CODE=${STATUS#* }
  case "$STATE" in
    stopped|destroyed|failed|missing) break ;;
  esac
  sleep "$POLL_SECS"
  waited=$((waited + POLL_SECS))
done
kill "$LOGS_PID" 2>/dev/null || true
LOGS_PID=""

(( waited < RESTORE_TIMEOUT_SECS )) || die "restore still running after ${RESTORE_TIMEOUT_SECS}s"
RESTORE_RUNNING=0
[[ "$EXIT_CODE" == "0" ]] || die "restore machine exited with code $EXIT_CODE (state $STATE)"
log "restore_from_r2.py exited 0"

# ---------------------------------------------------------------------------
# 3. Verify against the live database
# ---------------------------------------------------------------------------
log "Verifying"
wait_for_db || die "MariaDB is not answering after the restore"

echo "Row counts:"
db_sql "SELECT 'agreements', COUNT(*) FROM agreements UNION ALL SELECT 'sections', COUNT(*) FROM sections UNION ALL SELECT 'latest_sections_search', COUNT(*) FROM latest_sections_search UNION ALL SELECT 'section_text_search', COUNT(*) FROM section_text_search"

FT_INDEX=$(db_sql "SELECT COUNT(*) FROM information_schema.STATISTICS WHERE TABLE_SCHEMA='pdx' AND TABLE_NAME='section_text_search' AND INDEX_TYPE='FULLTEXT'" | tr -d '[:space:]')
[[ "$FT_INDEX" == "1" ]] || die "section_text_search has no FULLTEXT index (got '$FT_INDEX')"
echo "FULLTEXT index on section_text_search: present"

echo "Text search smoke query (sections matching 'material adverse effect'):"
db_sql "SELECT COUNT(*) FROM section_text_search WHERE MATCH(normalized_text) AGAINST('+material +adverse +effect' IN BOOLEAN MODE)"

echo "Recent MariaDB log lines:"
fly logs --app "$DB_APP" --no-tail 2>/dev/null | tail -n 15 || true

RESTORE_OK=1
