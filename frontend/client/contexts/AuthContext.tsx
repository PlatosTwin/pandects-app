import type { ReactNode } from "react";
import { createContext, useCallback, useEffect, useMemo, useState } from "react";
import type { AuthUser } from "@/lib/auth-types";
import { clearSessionToken, getSessionToken } from "@/lib/auth-session";
import { fetchMe, logoutSession } from "@/lib/auth-api";
import { authSessionTransport } from "@/lib/auth-transport";
import { isAuthWakeupError, withAuthWakeRetry } from "@/lib/auth-wake";

type AuthStatus = "loading" | "anonymous" | "authenticated";

interface AuthContextValue {
  status: AuthStatus;
  user: AuthUser | null;
  session_token: string | null;
  wakePending: boolean;
  refresh: () => Promise<void>;
  logout: () => void;
}

export const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const transport = authSessionTransport();
  const initialToken = transport === "bearer" ? getSessionToken() : null;
  const [status, setStatus] = useState<AuthStatus>(
    transport === "cookie" || initialToken ? "loading" : "anonymous",
  );
  const [user, setUser] = useState<AuthUser | null>(null);
  const [session_token, setSessionTokenState] = useState<string | null>(initialToken);
  const [wakePending, setWakePending] = useState(false);

  const refresh = useCallback(async () => {
    const token = transport === "bearer" ? getSessionToken() : null;
    setSessionTokenState(token);
    setStatus("loading");
    setWakePending(false);

    try {
      const me = await withAuthWakeRetry(async () => {
        try {
          return await fetchMe();
        } catch (error) {
          if (isAuthWakeupError(error)) {
            setWakePending(true);
          }
          throw error;
        }
      });
      setUser(me.user);
      setStatus("authenticated");
      setWakePending(false);
    } catch {
      if (transport === "bearer") {
        clearSessionToken();
      }
      setSessionTokenState(null);
      setUser(null);
      setStatus("anonymous");
      setWakePending(false);
    }
  }, [transport]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const logout = useCallback(() => {
    if (transport === "cookie") {
      void logoutSession().catch(() => undefined);
    } else {
      clearSessionToken();
    }
    setSessionTokenState(null);
    setUser(null);
    setStatus("anonymous");
    setWakePending(false);
  }, [transport]);

  const value = useMemo<AuthContextValue>(
    () => ({ status, user, session_token, wakePending, refresh, logout }),
    [status, user, session_token, wakePending, refresh, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
