import type { ReactNode } from "react";
import { createContext, useCallback, useEffect, useMemo, useState } from "react";
import type { AuthUser } from "@/lib/auth-types";
import { clearSessionToken, getSessionToken, setSessionToken } from "@/lib/auth-session";
import { fetchMe, logoutSession } from "@/lib/auth-api";
import { authSessionTransport } from "@/lib/auth-transport";

type AuthStatus = "loading" | "anonymous" | "authenticated";

interface AuthContextValue {
  status: AuthStatus;
  user: AuthUser | null;
  session_token: string | null;
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

  const refresh = useCallback(async () => {
    const token = transport === "bearer" ? getSessionToken() : null;
    setSessionTokenState(token);

    try {
      const me = await fetchMe();
      setUser(me.user);
      setStatus("authenticated");
    } catch {
      if (transport === "bearer") {
        clearSessionToken();
      }
      setSessionTokenState(null);
      setUser(null);
      setStatus("anonymous");
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
  }, [transport]);

  const value = useMemo<AuthContextValue>(
    () => ({ status, user, session_token, refresh, logout }),
    [status, user, session_token, refresh, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
