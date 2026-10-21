/**
 * How the OAuth consent screen should present the requesting application.
 *
 * A client's name is whatever it sent at registration, so it proves nothing.
 * The redirect URI's host is checked by the server against the client's
 * registration, so it is what the screen leads with.
 */

interface KnownClient {
  label: string;
  hosts: readonly string[];
  /** Names that claim to be this client. */
  namePattern: RegExp;
}

const KNOWN_CLIENTS: readonly KnownClient[] = [
  {
    label: "Claude",
    hosts: ["claude.ai", "claude.com"],
    namePattern: /claude|anthropic/i,
  },
  {
    label: "ChatGPT",
    hosts: ["chatgpt.com", "chat.openai.com"],
    namePattern: /chatgpt|openai/i,
  },
];

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

export type OAuthClientKind =
  /** Redirects to a known client's own host. */
  | "known"
  /** Redirects to a loopback address: an app on the user's computer. */
  | "local"
  /** Redirects to a host Pandects doesn't recognize. */
  | "unknown"
  /** Claims a known client's name but redirects somewhere else. */
  | "impersonation";

export interface OAuthClientIdentity {
  kind: OAuthClientKind;
  /** Host the browser returns to after consent (with port, if any). */
  host: string;
  /** For "known", the client; for "impersonation", the client it claims to be. */
  knownLabel: string | null;
}

function hostMatches(hostname: string, host: string): boolean {
  return hostname === host || hostname.endsWith(`.${host}`);
}

export function classifyOAuthClient(
  clientName: string | null,
  redirectUri: string,
): OAuthClientIdentity {
  let url: URL;
  try {
    url = new URL(redirectUri);
  } catch {
    return { kind: "unknown", host: redirectUri, knownLabel: null };
  }
  const hostname = url.hostname.toLowerCase();
  const host = url.host.toLowerCase();

  const byHost = KNOWN_CLIENTS.find(
    (client) =>
      url.protocol === "https:" &&
      client.hosts.some((known) => hostMatches(hostname, known)),
  );
  if (byHost) return { kind: "known", host, knownLabel: byHost.label };

  const claimed = clientName
    ? KNOWN_CLIENTS.find((client) => client.namePattern.test(clientName))
    : undefined;
  // Local apps (Claude Code, Codex, Cursor) legitimately use loopback, so a
  // known name on a loopback redirect is not treated as impersonation.
  if (LOOPBACK_HOSTS.has(hostname))
    return { kind: "local", host, knownLabel: null };
  if (claimed)
    return { kind: "impersonation", host, knownLabel: claimed.label };
  return { kind: "unknown", host, knownLabel: null };
}
