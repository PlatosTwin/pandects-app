export type AuthSessionTransport = "cookie" | "bearer";

export function authSessionTransport(): AuthSessionTransport {
  const raw = import.meta.env.VITE_AUTH_SESSION_TRANSPORT;
  if (raw === "cookie" || raw === "bearer") return raw;
  return import.meta.env.PROD ? "cookie" : "bearer";
}

