import type { ReactNode } from "react";
import { createContext, useCallback, useEffect, useMemo, useState } from "react";
import type { AuthUser } from "@/lib/auth-types";
import { clearSessionToken, getSessionToken, setSessionToken } from "@/lib/auth-session";
import {
  fetchMe,
  loginWithEmail,
  logoutSession,
  registerWithEmail,
  type LegalAcceptancePayload,
} from "@/lib/auth-api";
import { authSessionTransport } from "@/lib/auth-transport";

type AuthStatus = "loading" | "anonymous" | "authenticated";

interface AuthContextValue {
  status: AuthStatus;
  user: AuthUser | null;
  sessionToken: string | null;
  refresh: () => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  register: (
    email: string,
    password: string,
    legal: LegalAcceptancePayload,
    captchaToken?: string,
  ) => Promise<void>;
  logout: () => void;
}

export const AuthContext = createContext<AuthContextValue | null>(null);

function hasCsrfCookie(): boolean {
  if (typeof document === "undefined") return false;
  return document.cookie
    .split(";")
    .some((cookie) => cookie.trim().startsWith("pdcts_csrf="));
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const transport = authSessionTransport();
  const initialToken = transport === "bearer" ? getSessionToken() : null;
  const [status, setStatus] = useState<AuthStatus>(
    initialToken ? "loading" : "anonymous",
  );
  const [user, setUser] = useState<AuthUser | null>(null);
  const [sessionToken, setSessionTokenState] = useState<string | null>(initialToken);

  const refresh = useCallback(async () => {
    const token = transport === "bearer" ? getSessionToken() : null;
    setSessionTokenState(token);

    if (transport === "cookie" && !hasCsrfCookie()) {
      setUser(null);
      setStatus("anonymous");
      return;
    }

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

  const login = useCallback(async (email: string, password: string) => {
    const res = await loginWithEmail(email, password);
    if (transport === "bearer") {
      if (!res.sessionToken) {
        throw new Error("Missing session token.");
      }
      setSessionToken(res.sessionToken);
      setSessionTokenState(res.sessionToken);
    }
    setUser(res.user);
    setStatus("authenticated");
  }, [transport]);

  const register = useCallback(
    async (
      email: string,
      password: string,
      legal: LegalAcceptancePayload,
      captchaToken?: string,
    ) => {
      if (transport === "cookie") {
        void logoutSession().catch(() => undefined);
      } else {
        clearSessionToken();
      }
      setSessionTokenState(null);
      setUser(null);
      setStatus("anonymous");

      await registerWithEmail(email, password, legal, captchaToken);
    },
    [transport],
  );

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
    () => ({ status, user, sessionToken, refresh, login, register, logout }),
    [status, user, sessionToken, refresh, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
