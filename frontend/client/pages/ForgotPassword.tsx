import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/use-toast";
import { requestPasswordReset } from "@/lib/auth-api";

export default function ForgotPassword() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    try {
      await requestPasswordReset(email);
      setSent(true);
      toast({
        title: "Check your email",
        description: "If an account exists, we sent a reset link.",
      });
    } catch (err) {
      toast({
        title: "Couldn't send reset email",
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
          <h1 className="text-2xl font-semibold text-foreground">Forgot your password?</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Enter your email and we’ll send a reset link.
          </p>
        </div>
        <Card className="p-6 sm:p-8">
          {sent ? (
            <div className="grid gap-4 text-sm text-muted-foreground">
              <p>
                If the email matches an account, you’ll receive a password reset
                link shortly.
              </p>
              <Button
                className="w-full"
                onClick={() => navigate("/account", { replace: true })}
              >
                Back to sign in
              </Button>
            </div>
          ) : (
            <form className="grid gap-4 text-left" onSubmit={handleSubmit}>
              <div className="grid gap-2">
                <Label htmlFor="reset-email">Email</Label>
                <Input
                  id="reset-email"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  disabled={busy}
                />
              </div>
              <Button type="submit" disabled={busy} className="w-full">
                Send reset link
              </Button>
              <Link
                to="/account"
                className="text-center text-sm text-muted-foreground hover:text-foreground"
              >
                Back to sign in
              </Link>
            </form>
          )}
        </Card>
      </div>
    </PageShell>
  );
}
