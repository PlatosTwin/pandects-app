import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/use-toast";
import { resetPassword } from "@/lib/auth-api";

export default function ResetPassword() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const token = useMemo(() => params.get("token")?.trim() ?? "", [params]);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (password.length < 8) {
      toast({ title: "Password must be at least 8 characters." });
      return;
    }
    if (password !== confirm) {
      toast({ title: "Passwords do not match." });
      return;
    }
    setBusy(true);
    try {
      await resetPassword(token, password);
      toast({
        title: "Password updated",
        description: "You can now sign in with your new password.",
      });
      navigate("/account", { replace: true });
    } catch (err) {
      toast({
        title: "Couldn't reset password",
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <PageShell size="md">
      <div className="mx-auto w-full max-w-md text-center">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-foreground">Set a new password</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Choose a new password below.
          </p>
        </div>
        <Card className="p-6 sm:p-8">
          {!token ? (
            <div className="grid gap-4 text-sm text-muted-foreground" role="alert">
              <p>That reset link is missing or invalid.</p>
              <Link
                to="/auth/forgot-password"
                className="text-center text-sm text-foreground underline"
              >
                Request a new reset link
              </Link>
            </div>
          ) : (
            <form className="grid gap-4 text-left" onSubmit={handleSubmit}>
              <div className="grid gap-2">
                <Label htmlFor="new-password">New password</Label>
                <Input
                  id="new-password"
                  type="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={busy}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="confirm-password">Confirm password</Label>
                <Input
                  id="confirm-password"
                  type="password"
                  autoComplete="new-password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  required
                  disabled={busy}
                />
              </div>
              <Button type="submit" disabled={busy} className="w-full">
                Save new password
              </Button>
            </form>
          )}
        </Card>
      </div>
    </PageShell>
  );
}
