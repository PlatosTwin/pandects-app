export const FRONTEND_CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "base-uri 'self'",
  "object-src 'none'",
  "frame-ancestors 'none'",
  "script-src 'self' 'unsafe-inline' https://www.googletagmanager.com https://static.airtable.com https://accounts.google.com https://challenges.cloudflare.com",
  // connect-src and img-src are narrowed from `https:` so that an XSS can't
  // freely beacon stolen tokens to an attacker-controlled HTTPS endpoint.
  // Hosts here are the ones the SPA and its embedded analytics/captcha scripts
  // actually need; broaden cautiously if a legitimate request is blocked.
  "connect-src 'self' https://api.pandects.org https://www.googletagmanager.com https://www.google-analytics.com https://*.google-analytics.com https://challenges.cloudflare.com",
  "img-src 'self' data: https://www.googletagmanager.com https://www.google-analytics.com https://*.google-analytics.com",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' data: https://fonts.gstatic.com",
  "frame-src 'self' https://airtable.com https://accounts.google.com https://challenges.cloudflare.com",
  "form-action 'self'",
] as const;

export const FRONTEND_SECURITY_HEADERS = {
  "Content-Security-Policy": FRONTEND_CONTENT_SECURITY_POLICY.join("; "),
  "Cross-Origin-Opener-Policy": "same-origin-allow-popups",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Strict-Transport-Security": "max-age=15552000; includeSubDomains",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "SAMEORIGIN",
} as const;
