import { type ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "@/hooks/use-auth";
import { safeNextPath } from "@/lib/auth-next";

interface ProtectedRouteProps {
  children: ReactNode;
}

/**
 * Gates a route on an authenticated session.
 *
 * - status === "loading": renders children (pages own their own loading UI;
 *   that UI typically depends on user-derived state like wakePending).
 * - status === "anonymous": redirects to `/login?next=<current>` so the user
 *   lands back on the protected page after sign-in.
 * - otherwise: renders children.
 *
 * Pages no longer need their own `<Navigate>` guard.
 */
export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { status } = useAuth();
  const location = useLocation();

  if (status === "anonymous") {
    const nextPath = safeNextPath(`${location.pathname}${location.search}`);
    return (
      <Navigate to={`/login?next=${encodeURIComponent(nextPath)}`} replace />
    );
  }

  return <>{children}</>;
}
