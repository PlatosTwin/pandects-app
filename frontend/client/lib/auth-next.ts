export function safeNextPath(value: string | null | undefined): string {
  if (!value) return "/account";
  const trimmed = value.trim();
  if (!trimmed.startsWith("/")) return "/account";
  if (trimmed.startsWith("//")) return "/account";
  return trimmed;
}

export function buildAccountPathWithNext(nextPath: string): string {
  const safeNext = safeNextPath(nextPath);
  if (safeNext === "/account") return "/account";
  return `/account?next=${encodeURIComponent(safeNext)}`;
}
