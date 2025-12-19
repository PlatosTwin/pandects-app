import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


@dataclass(frozen=True)
class TemplateSpec:
    key: str
    resend_name: str
    html_path: Path
    variables: list[dict[str, str]]


def _load_env() -> None:
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
    load_dotenv()


def _templates_api_key() -> str:
    key = os.environ.get("RESEND_TEMPLATES_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "Missing RESEND_TEMPLATES_API_KEY. "
            "Create a Resend API key with Templates permission."
        )
    return key


def _resend_api_base_url() -> str:
    base = os.environ.get("RESEND_API_BASE_URL", "https://api.resend.com").strip().rstrip("/")
    if not base:
        raise SystemExit("Invalid RESEND_API_BASE_URL.")
    return base


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"Failed to read {path}: {e}") from e


def _render_for_resend(html: str) -> str:
    # Resend Templates use triple-brace handlebars variables, e.g. {{{PRODUCT}}}.
    return (
        html.replace("{{VERIFY_URL}}", "{{{VERIFY_URL}}}")
        .replace("{{YEAR}}", "{{{YEAR}}}")
        .replace("{{PREHEADER}}", "{{{PREHEADER}}}")
    )


def _request_json(
    *,
    method: str,
    path: str,
    query: dict[str, str] | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    base = _resend_api_base_url()
    url = f"{base}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"
    headers = {
        "Authorization": f"Bearer {_templates_api_key()}",
        "Accept": "application/json",
    }
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, method=method, headers=headers, data=body)
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Resend API error (HTTP {e.code}): {raw}") from e
    except URLError as e:
        raise SystemExit(f"Resend API network error: {e}") from e
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit(f"Resend API returned invalid JSON: {raw[:200]}...")
    return parsed if isinstance(parsed, dict) else {}


def _list_templates_by_name(*, name: str) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    after = None
    while True:
        query = {"limit": "100"}
        if after is not None:
            query["after"] = after
        payload = _request_json(method="GET", path="/templates", query=query)
        data = payload.get("data")
        if not isinstance(data, list):
            return matches
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("name") == name:
                matches.append(item)
        has_more = payload.get("has_more")
        if has_more is not True:
            return matches
        last_id = None
        for item in reversed(data):
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                last_id = item["id"]
                break
        if not last_id:
            return matches
        after = last_id


def _upsert_template(spec: TemplateSpec) -> None:
    html_raw = _read_text(spec.html_path)
    html = _render_for_resend(html_raw)

    existing = _list_templates_by_name(name=spec.resend_name)
    if existing:
        template_id = existing[0].get("id")
        if not isinstance(template_id, str) or not template_id.strip():
            raise SystemExit("Resend API returned a template without an id.")
        _request_json(
            method="PATCH",
            path=f"/templates/{template_id.strip()}",
            payload={"name": spec.resend_name, "html": html, "variables": spec.variables},
        )
        print(f"Updated template: {spec.resend_name} ({template_id})")
        return

    created = _request_json(
        method="POST",
        path="/templates",
        payload={"name": spec.resend_name, "html": html, "variables": spec.variables},
    )
    template_id = created.get("id")
    print(f"Created template: {spec.resend_name} ({template_id})")


def _templates() -> dict[str, TemplateSpec]:
    templates_dir = Path(__file__).with_name("email_templates")
    year = str(datetime.utcnow().year)
    return {
        "verify-email": TemplateSpec(
            key="verify-email",
            resend_name="pandects-verify-email",
            html_path=templates_dir / "verify_email.html",
            variables=[
                {
                    "key": "VERIFY_URL",
                    "type": "string",
                    "fallback_value": "https://pandects.org/account?emailVerified=1",
                },
                {"key": "YEAR", "type": "string", "fallback_value": year},
                {
                    "key": "PREHEADER",
                    "type": "string",
                    "fallback_value": "Verify your email to activate your Pandects account.",
                },
            ],
        )
    }


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(description="Sync email templates to Resend Templates.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sync = sub.add_parser("sync", help="Create/update templates in Resend.")
    sync.add_argument(
        "--only",
        action="append",
        default=[],
        help="Template key(s) to sync (e.g. --only verify-email).",
    )

    args = parser.parse_args()
    if args.cmd == "sync":
        specs = _templates()
        keys = args.only or list(specs.keys())
        for key in keys:
            spec = specs.get(key)
            if spec is None:
                raise SystemExit(f"Unknown template key: {key}")
            _upsert_template(spec)


if __name__ == "__main__":
    main()
