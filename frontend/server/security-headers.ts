import cspScriptHashes from "../shared/csp-script-hashes.generated.json" with { type: "json" };

// The SPA still ships a handful of inline <script> tags (gtag bootstrap +
// per-route JSON-LD). To drop 'unsafe-inline' from script-src we hash every
// inline-script body at build time and list the hashes here. Regenerate via
// `node scripts/compute-csp-script-hashes.mjs`; validate-csp.mjs guards against
// drift between this list and nginx.conf / netlify.toml.
const SCRIPT_HASHES: readonly string[] = (cspScriptHashes as { hashes: string[] }).hashes;

const SCRIPT_HASH_SOURCES = SCRIPT_HASHES.map((hash) => `'${hash}'`).join(" ");

export const FRONTEND_CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "base-uri 'self'",
  "object-src 'none'",
  "frame-ancestors 'none'",
  // script-src lists the exact inline-script SHA-256s plus the script-loading
  // origins we actually use. No 'unsafe-inline' — an XSS that injects a fresh
  // inline <script> is blocked.
  `script-src 'self' ${SCRIPT_HASH_SOURCES} https://www.googletagmanager.com https://static.airtable.com https://accounts.google.com https://challenges.cloudflare.com`,
  // connect-src / img-src / frame-src are narrowed to the hosts the SPA, its
  // embedded analytics/captcha scripts, and the OAuth/Airtable iframes actually
  // need. Broaden cautiously if a legitimate request is blocked — never to
  // a bare `https:` allowlist, which would let an XSS exfiltrate to any host.
  "connect-src 'self' https://api.pandects.org https://www.googletagmanager.com https://www.google-analytics.com https://*.google-analytics.com https://challenges.cloudflare.com",
  "img-src 'self' data: https://www.googletagmanager.com https://www.google-analytics.com https://*.google-analytics.com",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' data: https://fonts.gstatic.com",
  "frame-src 'self' https://airtable.com https://accounts.google.com https://challenges.cloudflare.com",
  "form-action 'self'",
  "upgrade-insecure-requests",
] as const;

export const FRONTEND_SECURITY_HEADERS = {
  "Content-Security-Policy": FRONTEND_CONTENT_SECURITY_POLICY.join("; "),
  "Cross-Origin-Opener-Policy": "same-origin-allow-popups",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=(), bluetooth=(), interest-cohort=()",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Strict-Transport-Security": "max-age=15552000; includeSubDomains",
  "X-Content-Type-Options": "nosniff",
  // Matches CSP frame-ancestors 'none' (which wins in modern browsers);
  // DENY keeps the legacy header consistent instead of contradicting it.
  "X-Frame-Options": "DENY",
} as const;
