import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { ChartContainer, ChartTooltip } from "@/components/ui/chart";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { toast } from "@/components/ui/use-toast";
import {
  createApiKey,
  deleteAccount,
  fetchUsage,
  listApiKeys,
  revokeApiKey,
  startZitadelWebsiteAuth,
} from "@/lib/auth-api";
import type { ApiKeySummary, UsageByDay, UsagePeriod } from "@/lib/auth-types";
import { apiUrl } from "@/lib/api-config";
import { cn } from "@/lib/utils";
import { Check, Copy } from "lucide-react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { trackEvent } from "@/lib/analytics";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import { formatDate } from "@/lib/format-utils";
import { safeNextPath } from "@/lib/auth-next";
import brandLinks from "@branding/links.json";

type UsageChartPoint = {
  day: string;
  count: number;
  cumulative: number;
};

const USAGE_RANGE_WINDOW_DAYS: Record<Exclude<UsagePeriod, "all">, number> = {
  "1w": 7,
  "1m": 30,
  "1y": 365,
};

const USAGE_RANGE_LABELS: Record<UsagePeriod, string> = {
  "1w": "1 week",
  "1m": "1 month",
  "1y": "1 year",
  all: "all time",
};

const usageDayTickFormatter = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});

const usageDayTickWithYearFormatter = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  year: "2-digit",
  timeZone: "UTC",
});

const usageDayTooltipFormatter = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});

const USAGE_DAY_MS = 24 * 60 * 60 * 1000;

function parseIsoDayToUtcMs(isoDay: string): number {
  const [year, month, day] = isoDay.split("-").map((part) => Number(part));
  return Date.UTC(year, month - 1, day);
}

function formatUsageDay(isoDay: string, includeYear: boolean): string {
  const date = new Date(`${isoDay}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return isoDay;
  return includeYear
    ? usageDayTickWithYearFormatter.format(date)
    : usageDayTickFormatter.format(date);
}

function formatUsageTooltipDay(isoDay: string): string {
  const date = new Date(`${isoDay}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return isoDay;
  return usageDayTooltipFormatter.format(date);
}

function utcDayToIso(utcMs: number): string {
  return new Date(utcMs).toISOString().slice(0, 10);
}

export default function Account() {
  const { status, user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const requestedNextPath = useMemo(
    () => safeNextPath(new URLSearchParams(location.search).get("next")),
    [location.search],
  );
  const [authPending, setAuthPending] = useState(false);
  const [apiKeysPending, setApiKeysPending] = useState(false);
  const [deletePending, setDeletePending] = useState(false);
  const accountForegroundRefreshInFlightRef = useRef(false);
  const accountForegroundRefreshAtMsRef = useRef(0);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (params.get("emailVerified") !== "1") return;
    toast({
      title: "Email verified!",
      description: "Sign in to create API keys and access full search results.",
    });
    params.delete("emailVerified");
    const nextQuery = params.toString();
    navigate(
      { pathname: location.pathname, search: nextQuery ? `?${nextQuery}` : "" },
      { replace: true },
    );
  }, [location.pathname, location.search, navigate]);

  const [apiKeys, setApiKeys] = useState<ApiKeySummary[]>([]);
  const [usageByDay, setUsageByDay] = useState<UsageByDay[]>([]);
  const [usageTotal, setUsageTotal] = useState(0);
  const [usagePeriod, setUsagePeriod] = useState<UsagePeriod>("1m");
  const [usageKeyFilter, setUsageKeyFilter] = useState("all");
  const [usageLoading, setUsageLoading] = useState(false);
  const usagePeriodRef = useRef<UsagePeriod>("1m");
  const usageKeyFilterRef = useRef("all");
  const usageRequestRunRef = useRef(0);
  const [accountDataLoading, setAccountDataLoading] = useState(false);
  const [accountDataLoaded, setAccountDataLoaded] = useState(false);
  const [accountDataError, setAccountDataError] = useState<string | null>(null);

  const [newKeyName, setNewKeyName] = useState("");
  const [revealedKey, setRevealedKey] = useState<string | null>(null);
  const [copiedNewKey, setCopiedNewKey] = useState(false);

  const [authBackendStatus, setAuthBackendStatus] = useState<
    "checking" | "ready" | "waking" | "failed"
  >("checking");
  const [authBackendError, setAuthBackendError] = useState<string | null>(null);

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");
  const [revokeDialogOpen, setRevokeDialogOpen] = useState(false);
  const [revokeTargetKey, setRevokeTargetKey] = useState<ApiKeySummary | null>(null);

  const hasAnyKey = apiKeys.some((k) => !k.revoked_at);

  const fetchWithTimeout = useCallback(
    async (input: RequestInfo | URL, init: RequestInit, timeoutMs: number) => {
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
      try {
        return await fetch(input, { ...init, signal: controller.signal });
      } finally {
        window.clearTimeout(timeoutId);
      }
    },
    [],
  );

  const pingAuthBackend = useCallback(async (): Promise<boolean> => {
    try {
      const res = await fetchWithTimeout(
        apiUrl("v1/auth/health"),
        { cache: "no-store" },
        5000,
      );
      return res.ok;
    } catch {
      return false;
    }
  }, [fetchWithTimeout]);

  const waitForAuthBackendReady = useCallback(async () => {
    setAuthBackendError(null);
    if (await pingAuthBackend()) {
      setAuthBackendStatus("ready");
      return;
    }
    setAuthBackendStatus("waking");
    const deadline = Date.now() + 60_000;
    const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
    const poll = async (): Promise<void> => {
      if (Date.now() >= deadline) {
        throw new Error("Authentication service is still initializing. Please retry in a moment.");
      }
      await delay(1000);
      if (await pingAuthBackend()) {
        setAuthBackendStatus("ready");
        return;
      }
      return poll();
    };
    await poll();
  }, [pingAuthBackend]);

  const fetchUsageForSelection = useCallback(
    (period: UsagePeriod, keyId: string) =>
      fetchUsage({
        period,
        apiKeyId: keyId === "all" ? undefined : keyId,
      }),
    [],
  );

  const loadUsageData = useCallback(
    async (
      { period, keyId, silent = false }: { period: UsagePeriod; keyId: string; silent?: boolean },
    ) => {
      const runId = ++usageRequestRunRef.current;
      if (!silent) setUsageLoading(true);
      try {
        const usage = await fetchUsageForSelection(period, keyId);
        if (usageRequestRunRef.current !== runId) return;
        setUsageByDay(usage.by_day);
        setUsageTotal(usage.total);
      } finally {
        if (!silent && usageRequestRunRef.current === runId) {
          setUsageLoading(false);
        }
      }
    },
    [fetchUsageForSelection],
  );

  const loadAccountData = useCallback(async ({ silent = false }: { silent?: boolean } = {}) => {
    if (!silent) setAccountDataLoading(true);
    try {
      const keys = await listApiKeys();
      setApiKeys(keys.keys);
      const selectedKeyExists =
        usageKeyFilterRef.current === "all"
          || keys.keys.some((key) => key.id === usageKeyFilterRef.current);
      const usageKey = selectedKeyExists ? usageKeyFilterRef.current : "all";
      if (usageKey !== usageKeyFilterRef.current) {
        usageKeyFilterRef.current = usageKey;
        setUsageKeyFilter(usageKey);
      }
      await loadUsageData({
        period: usagePeriodRef.current,
        keyId: usageKey,
        silent: true,
      });
      setAccountDataLoaded(true);
      setAccountDataError(null);
    } catch (err) {
      setAccountDataError(err instanceof Error ? err.message : String(err));
      throw err;
    } finally {
      if (!silent) setAccountDataLoading(false);
    }
  }, [loadUsageData]);

  useEffect(() => {
    usagePeriodRef.current = usagePeriod;
  }, [usagePeriod]);

  useEffect(() => {
    usageKeyFilterRef.current = usageKeyFilter;
  }, [usageKeyFilter]);

  useEffect(() => {
    if (!user) {
      setApiKeys([]);
      setUsageByDay([]);
      setUsageTotal(0);
      setUsagePeriod("1m");
      setUsageKeyFilter("all");
      usagePeriodRef.current = "1m";
      usageKeyFilterRef.current = "all";
      setUsageLoading(false);
      setAccountDataLoaded(false);
      setAccountDataLoading(false);
      setAccountDataError(null);
      return;
    }
    void loadAccountData().catch((err) => {
      toast({ title: "Failed to load account", description: String(err) });
    });
  }, [loadAccountData, user]);

  useEffect(() => {
    if (!user) return;

    const refreshAccountDataIfVisible = () => {
      if (document.visibilityState !== "visible") return;
      if (accountForegroundRefreshInFlightRef.current) return;
      const now = Date.now();
      if (now - accountForegroundRefreshAtMsRef.current < 1200) return;
      accountForegroundRefreshInFlightRef.current = true;
      accountForegroundRefreshAtMsRef.current = now;
      void loadAccountData({ silent: true })
        .catch(() => undefined)
        .finally(() => {
          accountForegroundRefreshInFlightRef.current = false;
        });
    };

    const onFocus = () => refreshAccountDataIfVisible();
    const onVisibilityChange = () => refreshAccountDataIfVisible();

    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [loadAccountData, user]);

  useEffect(() => {
    setCopiedNewKey(false);
  }, [revealedKey]);

  useEffect(() => {
    if (user) {
      setAuthBackendStatus("ready");
      setAuthBackendError(null);
      return;
    }
    setAuthBackendStatus("checking");
    void waitForAuthBackendReady().catch((err) => {
      setAuthBackendError(String(err));
      setAuthBackendStatus("failed");
    });
  }, [user, waitForAuthBackendReady]);

  const redactedReminder = useMemo(() => {
    if (status !== "authenticated") return null;
    if (hasAnyKey) return null;
    return "Create an API key to use the API programmatically.";
  }, [status, hasAnyKey]);

  const accountDataBootstrapping = !!user && !accountDataLoaded && !accountDataError;
  const docsUrl = import.meta.env.DEV ? "http://localhost:3001" : brandLinks.docsSiteUrl;
  const activeApiKeys = useMemo(() => apiKeys.filter((key) => !key.revoked_at), [apiKeys]);
  const revokedApiKeys = useMemo(() => apiKeys.filter((key) => !!key.revoked_at), [apiKeys]);
  const usageKeyOptions = useMemo(() => {
    return [
      { id: "all", label: "All keys" },
      ...apiKeys.map((key) => {
        const keyName = key.name?.trim() ? key.name.trim() : "Untitled key";
        return {
          id: key.id,
          label: key.revoked_at ? `${keyName} (${key.prefix}) [revoked]` : `${keyName} (${key.prefix})`,
        };
      }),
    ];
  }, [apiKeys]);
  const usageChartData = useMemo<UsageChartPoint[]>(() => {
    const byDayMap = new Map(usageByDay.map((row) => [row.day, Math.max(0, Number(row.count))]));
    const todayIso = new Date().toISOString().slice(0, 10);
    const todayMs = parseIsoDayToUtcMs(todayIso);

    const sortedDays = [...new Set(usageByDay.map((row) => row.day))].sort();
    if (usagePeriod === "all" && sortedDays.length === 0) return [];

    let startMs = todayMs;
    let endMs = todayMs;
    if (usagePeriod === "all") {
      startMs = parseIsoDayToUtcMs(sortedDays[0]);
      endMs = parseIsoDayToUtcMs(sortedDays[sortedDays.length - 1]);
    } else {
      startMs = todayMs - (USAGE_RANGE_WINDOW_DAYS[usagePeriod] - 1) * USAGE_DAY_MS;
    }

    if (endMs < startMs) return [];

    const points: UsageChartPoint[] = [];
    let cumulative = 0;
    for (let dayMs = startMs; dayMs <= endMs; dayMs += USAGE_DAY_MS) {
      const day = utcDayToIso(dayMs);
      const count = byDayMap.get(day) ?? 0;
      cumulative += count;
      points.push({ day, count, cumulative });
    }
    return points;
  }, [usageByDay, usagePeriod]);
  const usageXAxisIncludesYear = usagePeriod === "1y" || usagePeriod === "all";
  const selectedUsageKeyLabel = useMemo(
    () => usageKeyOptions.find((option) => option.id === usageKeyFilter)?.label ?? "All keys",
    [usageKeyFilter, usageKeyOptions],
  );

  const openRevokeDialog = useCallback((key: ApiKeySummary) => {
    setRevokeTargetKey(key);
    setRevokeDialogOpen(true);
  }, []);

  const confirmRevokeKey = useCallback(async () => {
    if (!revokeTargetKey) return;
    setApiKeysPending(true);
    try {
      await revokeApiKey(revokeTargetKey.id);
      await loadAccountData();
      setRevokeDialogOpen(false);
      setRevokeTargetKey(null);
      toast({ title: "API key revoked" });
    } catch (err) {
      toast({
        title: "Failed to revoke key",
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setApiKeysPending(false);
    }
  }, [loadAccountData, revokeTargetKey]);

  return (
    <PageShell
      title="Account"
      subtitle={
        <span className="text-sm text-muted-foreground">
          Sign in to unlock full access, manage API keys, and view API usage.
        </span>
      }
      size="md"
      actions={
        user ? (
          <Button variant="outline" onClick={logout}>
            Sign out
          </Button>
        ) : null
      }
    >
      {status === "loading" ? (
        <Card className="p-6" role="status" aria-live="polite">
          Loading…
        </Card>
      ) : !user ? (
        <Card className="relative p-6">
          {authBackendStatus !== "ready" ? (
            <div
              className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 rounded-lg bg-background/70 px-6 text-center"
              role={authBackendStatus === "failed" ? "alert" : "status"}
              aria-live={authBackendStatus === "failed" ? "assertive" : "polite"}
            >
              {authBackendStatus === "failed" ? (
                <>
                  <div className="text-sm font-medium">
                    Auth is unavailable right now
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {authBackendError ||
                      "The auth database may still be starting. Please retry."}
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => void waitForAuthBackendReady()}
                  >
                    Retry
                  </Button>
                </>
              ) : (
                <>
                  <LoadingSpinner size="md" aria-label="Initializing authentication service" />
                  <div className="text-sm font-medium">
                    Initializing authentication service…
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Sign-in and account creation will be available shortly.
                  </div>
                </>
              )}
            </div>
          ) : null}
          <div
            className={cn(
              "grid gap-6",
              authBackendStatus !== "ready" && "opacity-50",
            )}
            aria-busy={authBackendStatus !== "ready"}
          >
            <div className="grid gap-3">
              <div>
                <h2 className="text-xl font-semibold">Sign in with Pandects Auth</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Website sign-in and MCP access now use the same external auth identity.
                </p>
              </div>
              <Button
                disabled={authPending || authBackendStatus !== "ready"}
                onClick={() => {
                  setAuthPending(true);
                  void startZitadelWebsiteAuth(requestedNextPath, "google")
                    .then(({ authorize_url }) => {
                      window.location.assign(authorize_url);
                    })
                    .catch((err) => {
                      setAuthPending(false);
                      toast({
                        title: "Could not start sign-in",
                        description: err instanceof Error ? err.message : String(err),
                      });
                    });
                }}
                className="w-full sm:w-auto"
              >
                {authPending ? "Redirecting…" : "Continue with Google"}
              </Button>
              <Button
                variant="outline"
                disabled={authPending || authBackendStatus !== "ready"}
                onClick={() => {
                  setAuthPending(true);
                  void startZitadelWebsiteAuth(requestedNextPath, "email")
                    .then(({ authorize_url }) => {
                      window.location.assign(authorize_url);
                    })
                    .catch((err) => {
                      setAuthPending(false);
                      toast({
                        title: "Could not start sign-in",
                        description: err instanceof Error ? err.message : String(err),
                      });
                    });
                }}
                className="w-full sm:w-auto"
              >
                {authPending ? "Redirecting…" : "Continue with email"}
              </Button>
            </div>
            <div className="rounded-lg border border-border/60 bg-muted/20 p-4 text-sm text-muted-foreground">
              New signups, Google sign-in, and email sign-in are handled by the auth provider. After sign-in, Pandects still manages your API keys, legal acceptance, and app session.
            </div>
          </div>
        </Card>
      ) : (
        <div className="grid gap-6">
          {redactedReminder ? (
            <Alert>
              <AlertTitle>API access</AlertTitle>
              <AlertDescription>{redactedReminder}</AlertDescription>
            </Alert>
          ) : null}

          <Card className="p-6">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <h2 className="text-xl font-semibold">API keys</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Use `X-API-Key` for API access. Keep keys secret — you can view
                  a newly created key only once. Full API examples and reference are on the{" "}
                  <a
                    href={docsUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-primary hover:underline"
                  >
                    docs site
                  </a>
                  .
                </p>
              </div>
              <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:items-center">
                <Button
                  variant="outline"
                  onClick={() => {
                    setApiKeysPending(true);
                    loadAccountData()
                      .catch((err) => {
                        toast({ title: "Failed to refresh", description: String(err) });
                      })
                      .finally(() => setApiKeysPending(false));
                  }}
                  disabled={apiKeysPending || accountDataLoading || accountDataBootstrapping}
                  className="w-full sm:w-auto"
                >
                  Refresh
                </Button>
                <Label htmlFor="api-key-name" className="sr-only">
                  Key name (optional)
                </Label>
                <Input
                  id="api-key-name"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  placeholder="Key name (optional)"
                  className="w-full sm:w-48"
                  disabled={apiKeysPending || accountDataLoading || accountDataBootstrapping}
                />
                <Button
                  onClick={async () => {
                    trackEvent("api_key_new_click", { from_path: "/account" });
                    setApiKeysPending(true);
                    try {
                      const created = await createApiKey(newKeyName || undefined);
                      setRevealedKey(created.api_key_plaintext);
                      setNewKeyName("");
                      await loadAccountData();
                    } catch (err) {
                      toast({
                        title: "Failed to create API key",
                        description: String(err),
                      });
                    } finally {
                      setApiKeysPending(false);
                    }
                  }}
                  disabled={apiKeysPending || accountDataLoading || accountDataBootstrapping}
                  className="w-full sm:w-auto"
                >
                  New key
                </Button>
              </div>
            </div>

            {accountDataError ? (
              <Alert className="mt-4" variant="destructive">
                <AlertTitle>Account data unavailable</AlertTitle>
                <AlertDescription>{accountDataError}</AlertDescription>
              </Alert>
            ) : null}

            <div className="mt-4">
              <div className="grid gap-4 sm:hidden">
                {accountDataBootstrapping ? (
                  <div className="rounded-md border border-dashed border-border px-3 py-4 text-sm text-muted-foreground">
                    Loading API keys…
                  </div>
                ) : (
                  <>
                    <section className="grid gap-3">
                      <h3 className="text-sm font-medium">Active keys</h3>
                      {activeApiKeys.length === 0 ? (
                        <div className="rounded-md border border-dashed border-border px-3 py-4 text-sm text-muted-foreground">
                          No active API keys.
                        </div>
                      ) : (
                        activeApiKeys.map((k) => (
                          <div
                            key={k.id}
                            className="rounded-md border border-border/60 p-3"
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0">
                                <div className="text-sm font-medium text-foreground">
                                  {k.name ?? "Untitled key"}
                                </div>
                                <div className="mt-1 text-xs text-muted-foreground">
                                  Prefix
                                </div>
                                <div className="font-mono text-xs text-foreground">
                                  {k.prefix}
                                </div>
                              </div>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => openRevokeDialog(k)}
                                disabled={apiKeysPending}
                                className="shrink-0"
                              >
                                Revoke
                              </Button>
                            </div>
                            <div className="mt-3 grid gap-2 text-xs text-muted-foreground">
                              <div className="flex items-center justify-between gap-3">
                                <span>Created</span>
                                <span className="text-foreground">
                                  {formatDate(k.created_at)}
                                </span>
                              </div>
                              <div className="flex items-center justify-between gap-3">
                                <span>Last used</span>
                                <span className="text-foreground">
                                  {formatDate(k.last_used_at)}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))
                      )}
                    </section>
                    {revokedApiKeys.length > 0 ? (
                      <section className="grid gap-3">
                        <h3 className="text-sm font-medium">Revoked keys</h3>
                        {revokedApiKeys.map((k) => (
                          <div
                            key={k.id}
                            className="rounded-md border border-border/60 bg-muted/20 p-3"
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0">
                                <div className="text-sm font-medium text-foreground">
                                  {k.name ?? "Untitled key"}
                                </div>
                                <div className="mt-1 text-xs text-muted-foreground">
                                  Prefix
                                </div>
                                <div className="font-mono text-xs text-foreground">
                                  {k.prefix}
                                </div>
                              </div>
                              <div className="rounded border border-border px-2 py-1 text-xs text-muted-foreground">
                                Revoked
                              </div>
                            </div>
                            <div className="mt-3 grid gap-2 text-xs text-muted-foreground">
                              <div className="flex items-center justify-between gap-3">
                                <span>Created</span>
                                <span className="text-foreground">
                                  {formatDate(k.created_at)}
                                </span>
                              </div>
                              <div className="flex items-center justify-between gap-3">
                                <span>Last used</span>
                                <span className="text-foreground">
                                  {formatDate(k.last_used_at)}
                                </span>
                              </div>
                              <div className="flex items-center justify-between gap-3">
                                <span>Revoked</span>
                                <span className="text-foreground">
                                  {formatDate(k.revoked_at)}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </section>
                    ) : null}
                  </>
                )}
              </div>
              <div className="hidden sm:grid sm:gap-6">
                <section className="overflow-x-auto">
                  <h3 className="mb-2 text-sm font-medium">Active keys</h3>
                  <table className="w-full text-sm">
                    <caption className="sr-only">Active API keys</caption>
                    <thead className="text-muted-foreground">
                      <tr className="border-b">
                        <th scope="col" className="py-2 text-left font-medium">Name</th>
                        <th scope="col" className="py-2 text-left font-medium">Prefix</th>
                        <th scope="col" className="py-2 text-left font-medium">Created</th>
                        <th scope="col" className="py-2 text-left font-medium">Last used</th>
                        <th scope="col" className="py-2 text-right font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {accountDataBootstrapping ? (
                        <tr>
                          <td className="py-3 text-muted-foreground" colSpan={5}>
                            Loading API keys…
                          </td>
                        </tr>
                      ) : activeApiKeys.length === 0 ? (
                        <tr>
                          <td className="py-3 text-muted-foreground" colSpan={5}>
                            No active API keys.
                          </td>
                        </tr>
                      ) : (
                        activeApiKeys.map((k) => (
                          <tr key={k.id} className="border-b last:border-b-0">
                            <td className="py-3">{k.name ?? "—"}</td>
                            <td className="py-3 font-mono">{k.prefix}</td>
                            <td className="py-3">{formatDate(k.created_at)}</td>
                            <td className="py-3">{formatDate(k.last_used_at)}</td>
                            <td className="py-3 text-right">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => openRevokeDialog(k)}
                                disabled={apiKeysPending}
                              >
                                Revoke
                              </Button>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </section>
                {revokedApiKeys.length > 0 ? (
                  <section className="overflow-x-auto">
                    <h3 className="mb-2 text-sm font-medium">Revoked keys</h3>
                    <table className="w-full text-sm">
                      <caption className="sr-only">Revoked API keys</caption>
                      <thead className="text-muted-foreground">
                        <tr className="border-b">
                          <th scope="col" className="py-2 text-left font-medium">Name</th>
                          <th scope="col" className="py-2 text-left font-medium">Prefix</th>
                          <th scope="col" className="py-2 text-left font-medium">Created</th>
                          <th scope="col" className="py-2 text-left font-medium">Last used</th>
                          <th scope="col" className="py-2 text-left font-medium">Revoked</th>
                        </tr>
                      </thead>
                      <tbody>
                        {revokedApiKeys.map((k) => (
                          <tr key={k.id} className="border-b last:border-b-0">
                            <td className="py-3">{k.name ?? "—"}</td>
                            <td className="py-3 font-mono">{k.prefix}</td>
                            <td className="py-3">{formatDate(k.created_at)}</td>
                            <td className="py-3">{formatDate(k.last_used_at)}</td>
                            <td className="py-3">{formatDate(k.revoked_at)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </section>
                ) : null}
              </div>
            </div>
          </Card>

          <Card className="p-6 border-t border-border/60 pt-6 mt-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <h2 className="text-xl font-semibold">Usage</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Total in selected {USAGE_RANGE_LABELS[usagePeriod]} period:{" "}
                  {usageTotal.toLocaleString()} API requests.
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Showing: {selectedUsageKeyLabel}
                </p>
              </div>

              <div className="grid gap-3 sm:grid-cols-[auto_minmax(0,260px)] sm:items-end">
                <div className="flex flex-col gap-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Range
                  </span>
                  <ToggleGroup
                    type="single"
                    value={usagePeriod}
                    onValueChange={(value) => {
                      if (value !== "1w" && value !== "1m" && value !== "1y" && value !== "all") {
                        return;
                      }
                      if (value === usagePeriod) return;
                      setUsagePeriod(value);
                      usagePeriodRef.current = value;
                      void loadUsageData({
                        period: value,
                        keyId: usageKeyFilterRef.current,
                      }).catch((err) => {
                        toast({ title: "Failed to load usage", description: String(err) });
                      });
                    }}
                    variant="outline"
                    size="xs"
                    aria-label="Usage time range"
                    className="justify-start"
                  >
                    <ToggleGroupItem value="1w" disabled={accountDataBootstrapping}>
                      1W
                    </ToggleGroupItem>
                    <ToggleGroupItem value="1m" disabled={accountDataBootstrapping}>
                      1M
                    </ToggleGroupItem>
                    <ToggleGroupItem value="1y" disabled={accountDataBootstrapping}>
                      1Y
                    </ToggleGroupItem>
                    <ToggleGroupItem value="all" disabled={accountDataBootstrapping}>
                      All
                    </ToggleGroupItem>
                  </ToggleGroup>
                </div>

                <div className="flex flex-col gap-1">
                  <Label htmlFor="usage-key-filter" className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    API key
                  </Label>
                  <Select
                    value={usageKeyFilter}
                    onValueChange={(value) => {
                      if (value === usageKeyFilter) return;
                      setUsageKeyFilter(value);
                      usageKeyFilterRef.current = value;
                      void loadUsageData({
                        period: usagePeriodRef.current,
                        keyId: value,
                      }).catch((err) => {
                        toast({ title: "Failed to load usage", description: String(err) });
                      });
                    }}
                  >
                    <SelectTrigger id="usage-key-filter" className="h-8 w-full" disabled={accountDataBootstrapping}>
                      <SelectValue placeholder="All keys" />
                    </SelectTrigger>
                    <SelectContent>
                      {usageKeyOptions.map((option) => (
                        <SelectItem key={option.id} value={option.id}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>

            <div className="mt-4 rounded-lg border border-border/60 bg-muted/20 p-3">
              {accountDataBootstrapping ? (
                <div className="text-sm text-muted-foreground">Loading usage…</div>
              ) : usageChartData.length === 0 ? (
                <div className="text-sm text-muted-foreground">No usage yet.</div>
              ) : (
                <>
                  <ChartContainer
                    className="h-[260px] w-full min-w-0 aspect-auto sm:h-[300px]"
                    config={{
                      requests: {
                        label: "Requests",
                        color: "hsl(215 88% 56%)",
                      },
                    }}
                    role="img"
                    aria-label={`Usage chart for ${USAGE_RANGE_LABELS[usagePeriod]}, filtered to ${selectedUsageKeyLabel}.`}
                  >
                    <LineChart data={usageChartData} margin={{ top: 8, right: 20, left: 8, bottom: 0 }}>
                      <CartesianGrid vertical={false} />
                      <XAxis
                        dataKey="day"
                        tickFormatter={(value) =>
                          formatUsageDay(String(value), usageXAxisIncludesYear)
                        }
                        minTickGap={24}
                        tickMargin={8}
                      />
                      <YAxis
                        allowDecimals={false}
                        tickMargin={8}
                        width={48}
                        tickFormatter={(value) => Number(value).toLocaleString()}
                      />
                      <ChartTooltip
                        cursor={{ strokeDasharray: "4 4" }}
                        content={({ active, payload }) => {
                          if (!active || !payload?.length) return null;
                          const point = payload[0]?.payload as UsageChartPoint | undefined;
                          if (!point) return null;
                          return (
                            <div className="min-w-[180px] rounded-lg border border-border/50 bg-background px-3 py-2 text-xs shadow-xl">
                              <div className="mb-1.5 font-medium text-foreground">
                                {formatUsageTooltipDay(point.day)}
                              </div>
                              <div className="flex items-center justify-between gap-4">
                                <span className="text-muted-foreground">Day total</span>
                                <span className="font-mono tabular-nums text-foreground">
                                  {point.count.toLocaleString()}
                                </span>
                              </div>
                              <div className="mt-1 flex items-center justify-between gap-4">
                                <span className="text-muted-foreground">
                                  Cumulative ({USAGE_RANGE_LABELS[usagePeriod]})
                                </span>
                                <span className="font-mono tabular-nums text-foreground">
                                  {point.cumulative.toLocaleString()}
                                </span>
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="count"
                        name="Requests"
                        stroke="var(--color-requests)"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ChartContainer>
                </>
              )}
            </div>
          </Card>

          <Card className="p-6 border-t border-border/60 pt-6 mt-6">
            <h2 className="text-xl font-semibold text-destructive">
              Delete account
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              This permanently deletes your account and revokes your API keys.
              Usage data is retained.
            </p>

            <div className="mt-4">
              <Button
                variant="destructive"
                onClick={() => {
                  setDeleteConfirmText("");
                  setDeleteDialogOpen(true);
                }}
                disabled={deletePending}
              >
                Delete account
              </Button>
            </div>
          </Card>
        </div>
      )}

      <AlertDialog
        open={!!revealedKey}
        onOpenChange={(open) => {
          if (!open) setRevealedKey(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Your new API key</AlertDialogTitle>
            <AlertDialogDescription>
              Copy this now — you won’t be able to view it again.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="relative rounded-md border bg-muted px-3 py-2 pr-12 font-mono text-sm break-all">
            <button
              type="button"
              onClick={() => {
                if (!revealedKey) return;
                void navigator.clipboard.writeText(revealedKey);
                // Defer state and toast updates to avoid reflow
                setTimeout(() => {
                  setCopiedNewKey(true);
                  toast({ title: "Copied to clipboard" });
                }, 0);
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded bg-background p-1.5 shadow-sm border border-border/60 hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              title="Copy to clipboard"
              aria-label="Copy API key"
            >
              {copiedNewKey ? (
                <Check className="h-3 w-3 text-green-600" aria-hidden="true" />
              ) : (
                <Copy className="h-3 w-3 text-muted-foreground" aria-hidden="true" />
              )}
            </button>
            {revealedKey}
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel>Close</AlertDialogCancel>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={revokeDialogOpen}
        onOpenChange={(open) => {
          setRevokeDialogOpen(open);
          if (!open) setRevokeTargetKey(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Revoke API key?</AlertDialogTitle>
            <AlertDialogDescription>
              {revokeTargetKey ? (
                <>
                  This will immediately disable{" "}
                  <span className="font-medium">
                    {revokeTargetKey.name ?? `Key ${revokeTargetKey.prefix}`}
                  </span>
                  . This action cannot be undone.
                </>
              ) : (
                "This action cannot be undone."
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={apiKeysPending}>Cancel</AlertDialogCancel>
            <Button
              type="button"
              variant="destructive"
              disabled={apiKeysPending || !revokeTargetKey}
              onClick={() => void confirmRevokeKey()}
            >
              Revoke key
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete your account?</AlertDialogTitle>
            <AlertDialogDescription>
              Type <span className="font-mono">Delete</span> to confirm. This
              cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>

          <div className="grid gap-2">
            <Label htmlFor="delete-confirm">Confirmation</Label>
            <Input
              id="delete-confirm"
              value={deleteConfirmText}
              onChange={(e) => setDeleteConfirmText(e.target.value)}
              placeholder="Type Delete"
              autoComplete="off"
            />
          </div>

          <AlertDialogFooter>
            <AlertDialogCancel disabled={deletePending}>Cancel</AlertDialogCancel>
            <Button
              type="button"
              variant="destructive"
              disabled={deletePending || deleteConfirmText !== "Delete"}
              onClick={async () => {
                setDeletePending(true);
                try {
                  await deleteAccount({ confirm: deleteConfirmText });
                  toast({ title: "Account deleted" });
                  setDeleteDialogOpen(false);
                  logout();
                  navigate("/");
                } catch (err) {
                  toast({ title: "Delete failed", description: String(err) });
                } finally {
                  setDeletePending(false);
                }
              }}
            >
              Delete account
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </PageShell>
  );
}
