import type { ReactNode } from "react";
import { useAuth } from "@/hooks/use-auth";
import { LockdownNotice } from "@/components/LockdownNotice";

// TEMPORARY (July reopen): renders the wrapped data page only for signed-in
// (existing) accounts; everyone else gets the lockdown notice instead of the
// API errors a locked-down page would otherwise show. While auth is still
// resolving we render nothing so the page never fires its (403-bound) requests.
export function AccessGate({ children }: { children: ReactNode }) {
  const { status } = useAuth();
  if (status === "authenticated") {
    return <>{children}</>;
  }
  if (status === "loading") {
    return null;
  }
  return <LockdownNotice />;
}
