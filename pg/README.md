# Fly Postgres (Auth DB)

The backend uses a separate auth database (users, API keys, usage). In production on Fly, point it at your Fly Postgres cluster via `AUTH_DATABASE_URI` (or `DATABASE_URL`).

## Local development (recommended: Fly WireGuard)

`*.internal` / `*.flycast` hostnames are only reachable inside Fly’s private network. The most convenient way to access them from your laptop (without running `fly proxy` every time) is a Fly WireGuard tunnel.

### macOS note (avoid losing internet / DNS)

On macOS, importing a WireGuard config that sets a full-tunnel route (e.g. `AllowedIPs = 0.0.0.0/0` or `::/0`) and/or replaces system DNS can temporarily break normal Internet access. For this project you only want a split-tunnel to Fly’s private IPv6 network.

Recommended approach on macOS:

- Use the WireGuard macOS app to import the config.
- Ensure the peer uses only Fly private routing:
  - `AllowedIPs = fdaa::/16`
- Do **not** set `DNS = ...` in the tunnel config. Instead, add a per-domain resolver for Fly’s internal DNS so only `*.internal` lookups use it.

Per-domain resolver (macOS):

1. Create `/etc/resolver/internal`:

   `sudo mkdir -p /etc/resolver`

   `sudo sh -c 'cat > /etc/resolver/internal <<EOF
# Use the WireGuard-provided Fly DNS (from your `fly-wg.conf`).
# For this org it is typically:
nameserver fdaa:1f:c2e0::3
EOF'`

2. Flush DNS caches:

   `sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder`

Sanity checks:

- Normal Internet DNS should keep working while the tunnel is up (the tunnel should not set system DNS).
- Fly internal names should resolve: `dig pandects-pg.internal` (or `scutil --dns | rg resolver/internal` to confirm the resolver is active).

To undo that resolver later:

`sudo rm /etc/resolver/internal; sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder`

1. Create a WireGuard peer and save the config.

   Depending on your `flyctl` version, this may be either:

   `fly wireguard create -o ./fly-wg.conf`

   or positional args:

   `fly wireguard create <org> <region> <name> ./fly-wg.conf`

2. Bring the tunnel up:

   - macOS: import `fly-wg.conf` into the WireGuard app and toggle it on.
   - Linux (or macOS with `wireguard-tools` installed): `sudo wg-quick up ./fly-wg.conf`

3. Point the backend auth DB at the internal hostname (example):

   `AUTH_DATABASE_URI=postgresql://<user>:<password>@pandects-pg.internal:5432/<db>`

   If your unmanaged Postgres instance does not have TLS enabled, you may need:

   `AUTH_DATABASE_URI=postgresql://<user>:<password>@pandects-pg.internal:5432/<db>?sslmode=disable`

4. Initialize tables once (from the repo root):

   `python3 -m backend.init_auth_db`

5. Tear the tunnel down when done:

   - macOS: toggle the tunnel off in the WireGuard app.
   - `wg-quick`: `sudo wg-quick down ./fly-wg.conf`

## Local development (fallback: fly proxy)

If you don’t want to set up WireGuard, you can still connect via a local proxy:

1. `fly proxy 15432:5432 -a pandects-pg`
2. `AUTH_DATABASE_URI=postgresql://<user>:<password>@127.0.0.1:15432/<db>`
3. `python3 -m backend.init_auth_db`

## Fly deployment

- Ensure the app has a Postgres URL available as either `AUTH_DATABASE_URI` (preferred) or `DATABASE_URL` (common default when attaching).
- Deploy will run `python3 -m backend.init_auth_db` via `backend/fly.toml` release command.
