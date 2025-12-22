export type AuthSessionTransport = "cookie" | "bearer";

export function authSessionTransport(): AuthSessionTransport {
  return "cookie";
}
