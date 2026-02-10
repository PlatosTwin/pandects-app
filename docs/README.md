# Pandects Docs (Docusaurus)

Standalone Docusaurus app for Pandects Guides + API Reference.

## Local

```bash
cd docs
nvm use 20
npm run dev:clean
```

## Build

```bash
cd docs
npm run build
```

## Deploy (Fly)

```bash
cd docs
flyctl deploy -c fly.toml
```

This app uses:

- `auto_stop_machines = 'stop'`
- `auto_start_machines = true`
- `min_machines_running = 0`
