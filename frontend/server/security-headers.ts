export const FRONTEND_CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "base-uri 'self'",
  "object-src 'none'",
  "frame-ancestors 'none'",
  "script-src 'self' 'unsafe-inline' https://www.googletagmanager.com https://static.airtable.com https://accounts.google.com https://challenges.cloudflare.com",
  "connect-src 'self' https:",
  "img-src 'self' data: https:",
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
