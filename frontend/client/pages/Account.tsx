import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
import { ToastAction } from "@/components/ui/toast";
import {
  createApiKey,
  deleteAccount,
  fetchUsage,
  listApiKeys,
  loginWithGoogleCredential,
  resendVerificationEmail,
  revokeApiKey,
} from "@/lib/auth-api";
import type { ApiKeySummary, UsageByDay } from "@/lib/auth-types";
import { loadGoogleIdentityServices } from "@/lib/google-identity";
import { setSessionToken } from "@/lib/auth-session";
import { apiUrl } from "@/lib/api-config";
import { authSessionTransport } from "@/lib/auth-transport";
import { cn } from "@/lib/utils";
import { Check, Copy } from "lucide-react";
import { trackEvent } from "@/lib/analytics";
import { TurnstileWidget } from "@/components/TurnstileWidget";

function formatDate(value: string | null) {
  if (!value) return "—";
  const dt = new Date(value);
  return Number.isNaN(dt.getTime()) ? value : dt.toLocaleString();
}

export default function Account() {
  const { status, user, login, register, logout, refresh } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [googleStatus, setGoogleStatus] = useState<
    "loading" | "ready" | "unavailable"
  >("loading");
  const googleButtonRef = useRef<HTMLDivElement | null>(null);
  const refreshRef = useRef(refresh);
  const googleInitRunRef = useRef(0);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (params.get("emailVerified") !== "1") return;
    toast({
      title: "Email verified!",
      description: "Login to create API keys and access full search results.",
    });
    params.delete("emailVerified");
    const nextQuery = params.toString();
    navigate(
      { pathname: location.pathname, search: nextQuery ? `?${nextQuery}` : "" },
      { replace: true },
    );
  }, [location.pathname, location.search, navigate]);

  useEffect(() => {
    refreshRef.current = refresh;
  }, [refresh]);

  const [apiKeys, setApiKeys] = useState<ApiKeySummary[]>([]);
  const [usageByDay, setUsageByDay] = useState<UsageByDay[]>([]);
  const [usageTotal, setUsageTotal] = useState(0);

  const [newKeyName, setNewKeyName] = useState("");
  const [revealedKey, setRevealedKey] = useState<string | null>(null);
  const [copiedNewKey, setCopiedNewKey] = useState(false);

  const [legalAccepted, setLegalAccepted] = useState(false);
  const [legalCheckedAtMs, setLegalCheckedAtMs] = useState<number | null>(null);
  const [googlePendingCredential, setGooglePendingCredential] = useState<string | null>(null);
  const [googleNeedsLegal, setGoogleNeedsLegal] = useState(false);

  const [captchaEnabled, setCaptchaEnabled] = useState(false);
  const [captchaSiteKey, setCaptchaSiteKey] = useState<string | null>(null);
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const [captchaStatus, setCaptchaStatus] = useState<"loading" | "ready" | "unavailable">(
    "loading",
  );

  const [authBackendStatus, setAuthBackendStatus] = useState<
    "checking" | "ready" | "waking" | "failed"
  >("checking");
  const [authBackendError, setAuthBackendError] = useState<string | null>(null);

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");

  const hasAnyKey = apiKeys.some((k) => !k.revokedAt);
  const emailNotVerifiedMessage = "Email address not verified.";

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
        apiUrl("api/auth/health"),
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
        throw new Error("Auth database is still waking up. Please retry in a moment.");
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

  const loadAccountData = async () => {
    const [keys, usage] = await Promise.all([listApiKeys(), fetchUsage()]);
    setApiKeys(keys.keys);
    setUsageByDay(usage.byDay);
    setUsageTotal(usage.total);
  };

  useEffect(() => {
    if (!user) return;
    void loadAccountData().catch((err) => {
      toast({ title: "Failed to load account", description: String(err) });
    });
  }, [user]);

  useEffect(() => {
    if (!user) return;

    const refreshAccountData = () => {
      void loadAccountData().catch((err) => {
        toast({ title: "Failed to load account", description: String(err) });
      });
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") refreshAccountData();
    };

    window.addEventListener("focus", refreshAccountData);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.removeEventListener("focus", refreshAccountData);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [user]);

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

  useEffect(() => {
    if (user) return;
    if (authBackendStatus !== "ready") return;
    const runId = ++googleInitRunRef.current;
    setGoogleStatus("loading");
    setGooglePendingCredential(null);
    setGoogleNeedsLegal(false);

    const resolveClientId = async (): Promise<string | null> => {
      const fromEnv = import.meta.env.VITE_GOOGLE_OAUTH_CLIENT_ID;
      if (typeof fromEnv === "string" && fromEnv.trim().length > 0) {
        return fromEnv.trim();
      }
      try {
        const res = await fetch(apiUrl("api/auth/google/client-id"));
        if (!res.ok) return null;
        const data = (await res.json()) as { clientId?: unknown };
        return typeof data.clientId === "string" && data.clientId.trim().length > 0
          ? data.clientId.trim()
          : null;
      } catch {
        return null;
      }
    };

    void resolveClientId()
      .then(async (clientId) => {
        if (googleInitRunRef.current !== runId) return;
        if (!clientId) {
          setGoogleStatus("unavailable");
          return;
        }

        await loadGoogleIdentityServices();
        if (googleInitRunRef.current !== runId) return;
        if (!googleButtonRef.current) {
          setGoogleStatus("unavailable");
          return;
        }
        if (!window.google?.accounts?.id) {
          setGoogleStatus("unavailable");
          return;
        }

        window.google.accounts.id.initialize({
          client_id: clientId,
          callback: async ({ credential }) => {
            trackEvent("google_continue_click", { from_path: "/account" });
            setBusy(true);
            try {
              const res = await loginWithGoogleCredential(credential);
              if (authSessionTransport() === "bearer") {
                if (!res.sessionToken) throw new Error("Missing session token.");
                setSessionToken(res.sessionToken);
              }
              await refreshRef.current();
              toast({ title: "Signed in" });
            } catch (err) {
              const msg = String(err);
              if (msg.includes("legal_required")) {
                setGooglePendingCredential(credential);
                setGoogleNeedsLegal(true);
                toast({
                  title: "Agree to continue",
                  description: "Accept the Terms, Privacy Policy, and License to create your account.",
                });
              } else {
                toast({ title: "Google sign-in failed", description: msg });
              }
            } finally {
              setBusy(false);
            }
          },
          cancel_on_tap_outside: true,
        });

        googleButtonRef.current.replaceChildren();
        await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
        if (googleInitRunRef.current !== runId) return;
        const width = Math.round(
          Math.min(360, Math.max(240, googleButtonRef.current.getBoundingClientRect().width)),
        );
        window.google.accounts.id.renderButton(googleButtonRef.current, {
          theme: "outline",
          size: "large",
          text: "continue_with",
          shape: "rectangular",
          width,
        });

        setGoogleStatus("ready");
      })
      .catch(() => setGoogleStatus("unavailable"));
  }, [authBackendStatus, user]);

  useEffect(() => {
    if (user) return;
    if (authBackendStatus !== "ready") return;
    setCaptchaToken(null);
    setCaptchaStatus("loading");

    const fromEnv = import.meta.env.VITE_TURNSTILE_SITE_KEY;
    if (typeof fromEnv === "string" && fromEnv.trim().length > 0) {
      setCaptchaEnabled(true);
      setCaptchaSiteKey(fromEnv.trim());
      setCaptchaStatus("ready");
      return;
    }

    void fetch(apiUrl("api/auth/captcha/site-key"))
      .then(async (res) => {
        if (!res.ok) {
          setCaptchaEnabled(res.status === 503);
          setCaptchaSiteKey(null);
          setCaptchaStatus("unavailable");
          return;
        }
        const data = (await res.json()) as { enabled?: unknown; siteKey?: unknown };
        const enabled = data.enabled === true;
        const resolvedSiteKey =
          enabled && typeof data.siteKey === "string" && data.siteKey.trim()
            ? data.siteKey.trim()
            : null;
        setCaptchaEnabled(enabled);
        setCaptchaSiteKey(resolvedSiteKey);
        setCaptchaStatus(enabled && !resolvedSiteKey ? "unavailable" : "ready");
      })
      .catch(() => {
        setCaptchaEnabled(false);
        setCaptchaSiteKey(null);
        setCaptchaStatus("unavailable");
      });
  }, [authBackendStatus, user]);

  return (
    <PageShell
      title="Account"
      subtitle="Sign in to unlock full access, manage API keys, and view API usage."
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
        <Card className="p-6">Loading…</Card>
      ) : !user ? (
        <Card className="relative p-6">
          {authBackendStatus !== "ready" ? (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 rounded-lg bg-background/70 px-6 text-center">
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
                  <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                  <div className="text-sm font-medium">
                    Waking up the auth database…
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
          >
            <div className="flex justify-center pt-1">
              <div
                ref={googleButtonRef}
                className="flex min-h-[44px] w-full max-w-[360px] items-center justify-center overflow-visible px-1 py-1"
              />
            </div>
            {googleNeedsLegal ? (
              <div className="grid gap-3">
                <div className="flex items-start gap-3 rounded-lg border border-border/70 bg-muted/20 p-4 text-sm">
                  <Checkbox
                    id="legal-google"
                    checked={legalAccepted}
                    onCheckedChange={(next) => {
                      const isChecked = next === true;
                      setLegalAccepted(isChecked);
                      if (isChecked) {
                        const ts = Date.now();
                        setLegalCheckedAtMs(ts);
                        trackEvent("legal_consent_checked", {
                          context: "google_signup",
                          checked_at_ms: ts,
                        });
                      } else {
                        setLegalCheckedAtMs(null);
                      }
                    }}
                  />
                  <label htmlFor="legal-google" className="leading-relaxed">
                    I have read and agree to the{" "}
                    <Link
                      to="/terms"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "tos", context: "google_signup" })
                      }
                    >
                      Terms of Service
                    </Link>
                    ,{" "}
                    <Link
                      to="/privacy-policy"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "privacy", context: "google_signup" })
                      }
                    >
                      Privacy Policy
                    </Link>
                    , and{" "}
                    <Link
                      to="/license"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "license", context: "google_signup" })
                      }
                    >
                      License
                    </Link>
                    .
                  </label>
                </div>
                <Button
                  disabled={busy || !legalAccepted || !legalCheckedAtMs || !googlePendingCredential}
                  onClick={async () => {
                    if (!googlePendingCredential || !legalCheckedAtMs) return;
                    setBusy(true);
                    try {
                      trackEvent("legal_consent_submitted", {
                        context: "google_signup",
                        checked_at_ms: legalCheckedAtMs,
                        submitted_at_ms: Date.now(),
                      });
                      const res = await loginWithGoogleCredential(googlePendingCredential, {
                        checkedAtMs: legalCheckedAtMs,
                        docs: ["tos", "privacy", "license"],
                      });
                      if (authSessionTransport() === "bearer") {
                        if (!res.sessionToken) throw new Error("Missing session token.");
                        setSessionToken(res.sessionToken);
                      }
                      await refresh();
                      toast({ title: "Signed in" });
                      setGoogleNeedsLegal(false);
                      setGooglePendingCredential(null);
                    } catch (err) {
                      toast({ title: "Google sign-in failed", description: String(err) });
                    } finally {
                      setBusy(false);
                    }
                  }}
                  className="w-64 justify-self-center"
                >
                  Continue
                </Button>
              </div>
            ) : null}
            {googleStatus === "ready" ? (
              <div className="flex items-center gap-3 py-1">
                <div className="h-px flex-1 bg-border" />
                <div className="text-xs text-muted-foreground">or</div>
                <div className="h-px flex-1 bg-border" />
              </div>
            ) : googleStatus === "unavailable" ? (
              <div className="text-center text-xs text-muted-foreground">
                Google sign-in is not configured.
              </div>
            ) : (
              <div className="text-center text-xs text-muted-foreground">
                Loading Google sign-in…
              </div>
            )}
          </div>

            <Tabs defaultValue="signin" className="mt-6">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="signin">Sign in</TabsTrigger>
                <TabsTrigger value="register">Create account</TabsTrigger>
              </TabsList>

            <TabsContent value="signin">
              <form
                className="mt-4 grid gap-4"
                onSubmit={async (e) => {
                  e.preventDefault();
                  trackEvent("signin_click", { from_path: "/account", method: "email" });
                  setBusy(true);
                  try {
                    await waitForAuthBackendReady();
                    await login(email, password);
                    toast({ title: "Signed in" });
                  } catch (err) {
                    const message = err instanceof Error ? err.message : String(err);
                    if (message.includes(emailNotVerifiedMessage)) {
                      toast({
                        title: "Verify your email to sign in",
                        description:
                          "We sent a verification email when you signed up. Please check your inbox and spam folder.",
                        action: (
                          <ToastAction
                            altText="Resend verification email"
                            onClick={async () => {
                              try {
                                await resendVerificationEmail(email);
                                toast({
                                  title: "Verification email resent",
                                  description: "Check your inbox for the latest link.",
                                });
                              } catch (sendError) {
                                toast({
                                  title: "Couldn't resend email",
                                  description: String(sendError),
                                });
                              }
                            }}
                          >
                            Resend email
                          </ToastAction>
                        ),
                      });
                    } else {
                      toast({ title: "Sign-in failed", description: message });
                    }
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                <div className="grid gap-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    disabled={busy || authBackendStatus !== "ready"}
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="password">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    disabled={busy || authBackendStatus !== "ready"}
                  />
                </div>
                <Button
                  type="submit"
                  disabled={busy || authBackendStatus !== "ready"}
                  className="w-64 justify-self-center"
                >
                  Sign in
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="register">
              <form
                className="mt-4 grid gap-4"
                onSubmit={async (e) => {
                  e.preventDefault();
                  trackEvent("create_account_click", { from_path: "/account", method: "email" });
                  setBusy(true);
                  try {
                    await waitForAuthBackendReady();
                    if (!legalCheckedAtMs) {
                      throw new Error("Please accept the Terms, Privacy Policy, and License.");
                    }
                    if (captchaEnabled && !captchaToken) {
                      throw new Error("Please complete the captcha.");
                    }
                    trackEvent("legal_consent_submitted", {
                      context: "email_register",
                      checked_at_ms: legalCheckedAtMs,
                      submitted_at_ms: Date.now(),
                    });
                    await register(
                      email,
                      password,
                      {
                        checkedAtMs: legalCheckedAtMs,
                        docs: ["tos", "privacy", "license"],
                      },
                      captchaToken ?? undefined,
                    );
                    toast({
                      title: "Check your email",
                      description: "Verify your email address to finish creating your account.",
                    });
                    navigate("/");
                  } catch (err) {
                    toast({
                      title: "Registration failed",
                      description: String(err),
                    });
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                <div className="grid gap-2">
                  <Label htmlFor="email2">Email</Label>
                  <Input
                    id="email2"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    disabled={busy || authBackendStatus !== "ready"}
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="password2">Password</Label>
                  <Input
                    id="password2"
                    type="password"
                    autoComplete="new-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    disabled={busy || authBackendStatus !== "ready"}
                  />
                </div>
                {captchaEnabled ? (
                  captchaSiteKey ? (
                    <TurnstileWidget
                      siteKey={captchaSiteKey}
                      onToken={(token) => setCaptchaToken(token)}
                      onError={(message) =>
                        toast({ title: "Captcha error", description: message })
                      }
                    />
                  ) : captchaStatus === "unavailable" ? (
                    <div className="text-xs text-muted-foreground">
                      Captcha is temporarily unavailable.
                    </div>
                  ) : (
                    <div className="text-xs text-muted-foreground">Loading captcha…</div>
                  )
                ) : null}
                <div className="mt-2 flex items-start gap-3 rounded-lg border border-border/70 bg-muted/20 p-4 text-sm">
                  <Checkbox
                    id="legal-register"
                    checked={legalAccepted}
                    disabled={busy || authBackendStatus !== "ready"}
                    onCheckedChange={(next) => {
                      const isChecked = next === true;
                      setLegalAccepted(isChecked);
                      if (isChecked) {
                        const ts = Date.now();
                        setLegalCheckedAtMs(ts);
                        trackEvent("legal_consent_checked", {
                          context: "email_register",
                          checked_at_ms: ts,
                        });
                      } else {
                        setLegalCheckedAtMs(null);
                      }
                    }}
                  />
                  <label htmlFor="legal-register" className="leading-relaxed">
                    I have read and agree to the{" "}
                    <Link
                      to="/terms"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "tos", context: "email_register" })
                      }
                    >
                      Terms of Service
                    </Link>
                    ,{" "}
                    <Link
                      to="/privacy-policy"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "privacy", context: "email_register" })
                      }
                    >
                      Privacy Policy
                    </Link>
                    , and{" "}
                    <Link
                      to="/license"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "license", context: "email_register" })
                      }
                    >
                      License
                    </Link>
                    .
                  </label>
                </div>
                <Button
                  type="submit"
                  disabled={
                    busy ||
                    authBackendStatus !== "ready" ||
                    !legalAccepted ||
                    (captchaEnabled && !captchaToken)
                  }
                  className="w-64 justify-self-center"
                >
                  Create account
                </Button>
              </form>
            </TabsContent>
          </Tabs>
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
                <h2 className="text-lg font-semibold">API keys</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Use `X-API-Key` for API access. Keep keys secret — you can view
                  a newly created key only once.
                </p>
              </div>
              <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:items-center">
                <Button
                  variant="outline"
                  onClick={() => {
                    setBusy(true);
                    loadAccountData()
                      .catch((err) => {
                        toast({ title: "Failed to refresh", description: String(err) });
                      })
                      .finally(() => setBusy(false));
                  }}
                  disabled={busy}
                  className="w-full sm:w-auto"
                >
                  Refresh
                </Button>
                <Input
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  placeholder="Key name (optional)"
                  className="w-full sm:w-48"
                />
                <Button
                  onClick={async () => {
                    trackEvent("api_key_new_click", { from_path: "/account" });
                    setBusy(true);
                    try {
                      const created = await createApiKey(newKeyName || undefined);
                      setRevealedKey(created.apiKeyPlaintext);
                      setNewKeyName("");
                      await loadAccountData();
                    } catch (err) {
                      toast({
                        title: "Failed to create API key",
                        description: String(err),
                      });
                    } finally {
                      setBusy(false);
                    }
                  }}
                  disabled={busy}
                  className="w-full sm:w-auto"
                >
                  New key
                </Button>
              </div>
            </div>

            <div className="mt-4">
              <div className="grid gap-3 sm:hidden">
                {apiKeys.length === 0 ? (
                  <div className="rounded-md border border-dashed border-border px-3 py-4 text-sm text-muted-foreground">
                    No API keys yet.
                  </div>
                ) : (
                  apiKeys.map((k) => (
                    <div
                      key={k.id}
                      className="rounded-md border border-border p-3"
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
                          onClick={async () => {
                            setBusy(true);
                            try {
                              await revokeApiKey(k.id);
                              await loadAccountData();
                            } catch (err) {
                              toast({
                                title: "Failed to revoke key",
                                description: String(err),
                              });
                            } finally {
                              setBusy(false);
                            }
                          }}
                          disabled={busy || !!k.revokedAt}
                          className="shrink-0"
                        >
                          {k.revokedAt ? "Revoked" : "Revoke"}
                        </Button>
                      </div>
                      <div className="mt-3 grid gap-2 text-xs text-muted-foreground">
                        <div className="flex items-center justify-between gap-3">
                          <span>Created</span>
                          <span className="text-foreground">
                            {formatDate(k.createdAt)}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <span>Last used</span>
                          <span className="text-foreground">
                            {formatDate(k.lastUsedAt)}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
              <div className="hidden overflow-x-auto sm:block">
                <table className="w-full text-sm">
                  <thead className="text-muted-foreground">
                    <tr className="border-b">
                      <th className="py-2 text-left font-medium">Name</th>
                      <th className="py-2 text-left font-medium">Prefix</th>
                      <th className="py-2 text-left font-medium">Created</th>
                      <th className="py-2 text-left font-medium">Last used</th>
                      <th className="py-2 text-right font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {apiKeys.length === 0 ? (
                      <tr>
                        <td className="py-3 text-muted-foreground" colSpan={5}>
                          No API keys yet.
                        </td>
                      </tr>
                    ) : (
                      apiKeys.map((k) => (
                        <tr key={k.id} className="border-b last:border-b-0">
                          <td className="py-3">{k.name ?? "—"}</td>
                          <td className="py-3 font-mono">{k.prefix}</td>
                          <td className="py-3">{formatDate(k.createdAt)}</td>
                          <td className="py-3">{formatDate(k.lastUsedAt)}</td>
                          <td className="py-3 text-right">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={async () => {
                                setBusy(true);
                                try {
                                  await revokeApiKey(k.id);
                                  await loadAccountData();
                                } catch (err) {
                                  toast({
                                    title: "Failed to revoke key",
                                    description: String(err),
                                  });
                                } finally {
                                  setBusy(false);
                                }
                              }}
                              disabled={busy || !!k.revokedAt}
                            >
                              {k.revokedAt ? "Revoked" : "Revoke"}
                            </Button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h2 className="text-lg font-semibold">Usage (last 30 days)</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Total: {usageTotal.toLocaleString()} API requests.
            </p>

            <div className="mt-4 grid gap-2">
              {usageByDay.length === 0 ? (
                <div className="text-sm text-muted-foreground">No usage yet.</div>
              ) : (
                usageByDay.map((row) => (
                  <div
                    key={row.day}
                    className="flex flex-col gap-1 rounded-md border border-border px-3 py-2 text-sm sm:flex-row sm:items-center sm:justify-between"
                  >
                    <span className="font-mono text-xs text-muted-foreground">
                      {row.day}
                    </span>
                    <span className="text-sm font-medium text-foreground">
                      {row.count.toLocaleString()}
                    </span>
                  </div>
                ))
              )}
            </div>
          </Card>

          <Card className="p-6">
            <h2 className="text-lg font-semibold text-destructive">
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
                disabled={busy}
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
                setCopiedNewKey(true);
                toast({ title: "Copied to clipboard" });
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded bg-background p-1.5 shadow-sm border border-border hover:bg-accent"
              title="Copy to clipboard"
              aria-label="Copy API key"
            >
              {copiedNewKey ? (
                <Check className="h-3 w-3 text-green-600" />
              ) : (
                <Copy className="h-3 w-3 text-muted-foreground" />
              )}
            </button>
            {revealedKey}
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel>Close</AlertDialogCancel>
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
            <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
            <Button
              type="button"
              variant="destructive"
              disabled={busy || deleteConfirmText !== "Delete"}
              onClick={async () => {
                setBusy(true);
                try {
                  await deleteAccount({ confirm: deleteConfirmText });
                  toast({ title: "Account deleted" });
                  setDeleteDialogOpen(false);
                  logout();
                  navigate("/");
                } catch (err) {
                  toast({ title: "Delete failed", description: String(err) });
                } finally {
                  setBusy(false);
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
