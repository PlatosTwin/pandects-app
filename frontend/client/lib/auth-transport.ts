export type AuthSessionTransport = "cookie" | "bearer";

export function authSessionTransport(): AuthSessionTransport {
  const configured = import.meta.env.VITE_AUTH_SESSION_TRANSPORT;
  if (typeof configured === "string") {
    const normalized = configured.trim().toLowerCase();
    if (normalized === "cookie" || normalized === "bearer") {
      return normalized;
    }
  }
  return "cookie";
}
