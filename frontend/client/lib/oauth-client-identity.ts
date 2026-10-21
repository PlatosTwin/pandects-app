/**
 * How the OAuth consent screen should present the requesting application.
 *
 * A client's name is whatever it sent at registration, so it proves nothing.
 * Its registered redirect URIs decide where codes can go, and a grant covers
 * all of them, so a client is only named as a known application when every
 * one of them is that application's own host.
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
    namePattern: /\b(claude|anthropic)\b/i,
  },
  {
    label: "ChatGPT",
    hosts: ["chatgpt.com", "chat.openai.com"],
    namePattern: /\b(chatgpt|openai)\b/i,
  },
];

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

export type OAuthClientKind =
  /** Every redirect goes to one known client's own host. */
  | "known"
  /** Every redirect goes to a loopback address: an app on the user's computer. */
  | "local"
  /** Pandects doesn't recognize where the redirects go. */
  | "unknown"
  /** Claims a known client's name but can redirect somewhere else. */
  | "impersonation";

export interface OAuthClientIdentity {
  kind: OAuthClientKind;
  /** Host the browser returns to after consent (with port, if any). */
  host: string;
  /** Other registered hosts that can also receive a code once allowed. */
  otherHosts: string[];
  /** For "known", the client; for "impersonation", the client it claims to be. */
  knownLabel: string | null;
  /** For "impersonation", the registered hosts the claimed client doesn't use. */
  foreignHosts: string[];
}

interface ParsedRedirect {
  host: string;
  hostname: string;
  https: boolean;
}

function parseRedirect(uri: string): ParsedRedirect | null {
  try {
    const url = new URL(uri);
    return {
      host: url.host.toLowerCase(),
      // A trailing dot names the same host.
      hostname: url.hostname.toLowerCase().replace(/\.$/, ""),
      https: url.protocol === "https:",
    };
  } catch {
    return null;
  }
}

function knownClientFor(redirect: ParsedRedirect): KnownClient | undefined {
  if (!redirect.https) return undefined;
  return KNOWN_CLIENTS.find((client) =>
    client.hosts.includes(redirect.hostname),
  );
}

export function classifyOAuthClient(
  clientName: string | null,
  redirectUri: string,
  registeredRedirectUris: readonly string[],
): OAuthClientIdentity {
  const requested = parseRedirect(redirectUri);
  const host = requested?.host ?? redirectUri;
  const registered = registeredRedirectUris.map((uri) => ({
    uri,
    parsed: parseRedirect(uri),
  }));
  // Loopback redirects match on any port (RFC 8252), so compare their hostnames.
  const hostKey = (p: ParsedRedirect | null, raw: string) =>
    p === null ? raw : LOOPBACK_HOSTS.has(p.hostname) ? p.hostname : p.host;
  const requestedKey = hostKey(requested, redirectUri);
  const otherHosts = [
    ...new Map(
      registered
        .filter((entry) => hostKey(entry.parsed, entry.uri) !== requestedKey)
        .map((entry) => [
          hostKey(entry.parsed, entry.uri),
          entry.parsed?.host ?? entry.uri,
        ]),
    ).values(),
  ];
  const base = { host, otherHosts, knownLabel: null, foreignHosts: [] };

  const parsed = registered.map((entry) => entry.parsed);
  if (parsed.length === 0 || parsed.some((p) => p === null)) {
    return { ...base, kind: "unknown" };
  }
  const valid = parsed as ParsedRedirect[];

  const owners = new Set(valid.map(knownClientFor));
  const [owner] = owners;
  if (owners.size === 1 && owner) {
    return { ...base, kind: "known", knownLabel: owner.label };
  }

  // Local apps (Claude Code, Codex, Cursor) legitimately use loopback, so a
  // known name on loopback-only redirects is not treated as impersonation.
  if (valid.every((p) => LOOPBACK_HOSTS.has(p.hostname))) {
    return { ...base, kind: "local" };
  }

  const claimed = clientName
    ? KNOWN_CLIENTS.find((client) => client.namePattern.test(clientName))
    : undefined;
  if (claimed) {
    const foreignHosts = [
      ...new Set(
        valid.filter((p) => knownClientFor(p) !== claimed).map((p) => p.host),
      ),
    ];
    return {
      ...base,
      kind: "impersonation",
      knownLabel: claimed.label,
      foreignHosts,
    };
  }
  return { ...base, kind: "unknown" };
}
