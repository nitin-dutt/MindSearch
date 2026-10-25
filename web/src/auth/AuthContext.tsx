import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getMe, loadToken, login as apiLogin, setToken, setUnauthorizedHandler } from "../api/client";
import type { Me } from "../api/types";
import { DEV_USERS, defaultDevUser, type DevUser } from "../devAuth";

interface AuthState {
  user: Me | null;
  /** true while we check whether a stored session is still valid (so the login screen does not flash) */
  loading: boolean;
  /** shown on the login screen after the session ended on its own */
  notice: string | null;
  signIn: (username: string, password: string) => Promise<void>;
  signOut: () => void;
  /** local development only: the demo accounts that can be switched to (empty in a real build) */
  devUsers: DevUser[];
  switchTo: (username: string) => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<Me | null>(null);
  const [loading, setLoading] = useState<boolean>(() => loadToken() !== null || DEV_USERS.length > 0);
  const [notice, setNotice] = useState<string | null>(null);

  const signOut = useCallback(() => {
    setToken(null);
    setUser(null);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(() => {
      if (DEV_USERS.length > 0) {
        setToken(null);
        window.location.reload(); // development: sign in again as the default demo account instead of showing the login screen
        return;
      }
      setUser(null);
      setNotice("Your session ended. Please sign in again.");
    });
    const fallback = defaultDevUser(); // null unless the local-development shortcut is enabled
    const start = loadToken() === null ? Promise.reject(new Error("no session")) : getMe();
    start
      .then(setUser)
      .catch(async () => {
        setToken(null);
        if (!fallback) return;
        try {
          await apiLogin(fallback.username, fallback.password);
          setUser(await getMe());
        } catch {
          setToken(null); // the API is down or the demo account is gone: show the normal login screen
        }
      })
      .finally(() => setLoading(false));
  }, []);

  const signIn = useCallback(async (username: string, password: string) => {
    await apiLogin(username, password);
    setUser(await getMe());
    setNotice(null);
  }, []);

  // Development only: log in as another demo account and reload, so every screen and the chat start fresh for that account.
  const switchTo = useCallback(async (username: string) => {
    const target = DEV_USERS.find((u) => u.username === username);
    if (!target) return;
    await apiLogin(target.username, target.password);
    window.location.reload();
  }, []);

  const value = useMemo(
    () => ({ user, loading, notice, signIn, signOut, devUsers: DEV_USERS, switchTo }),
    [user, loading, notice, signIn, signOut, switchTo],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
