import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { verifyEmail } from "@/lib/auth-api";

export default function VerifyEmail() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);

  const token = useMemo(() => {
    const hashParams = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    const fromHash = hashParams.get("token");
    if (fromHash && fromHash.trim()) return fromHash.trim();
    return params.get("token")?.trim() ?? "";
  }, [params]);

  useEffect(() => {
    if (!window.location.hash) return;
    window.history.replaceState(
      null,
      document.title,
      window.location.pathname + window.location.search,
    );
  }, []);

  useEffect(() => {
    if (!token) {
      setError("missing_token");
      setBusy(false);
      return;
    }
    setBusy(true);
    setError(null);
    void verifyEmail(token)
      .then(() => {
        navigate("/account?emailVerified=1", { replace: true });
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        setBusy(false);
      });
  }, [navigate, token]);

  return (
    <PageShell title="Verifying email…" subtitle="Finishing email verification." size="md">
      <Card className="p-6">
        {error ? (
          <div className="grid gap-4" role="alert">
            <div>
              <div className="text-sm font-medium">Email verification failed</div>
              <div className="mt-1 text-sm text-muted-foreground">
                Error: {error}
              </div>
            </div>
            <Button asChild>
              <Link to="/account">Back to account</Link>
            </Button>
          </div>
        ) : busy ? (
          <div className="text-sm text-muted-foreground" role="status" aria-live="polite">
            Finishing verification…
          </div>
        ) : (
          <Button asChild>
            <Link to="/account">Back to account</Link>
          </Button>
        )}
      </Card>
    </PageShell>
  );
}
