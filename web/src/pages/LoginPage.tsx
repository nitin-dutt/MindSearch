import { useState, type FormEvent } from "react";
import { Eye, EyeOff, Lock } from "lucide-react";
import { ApiError } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { Brand } from "../components/AppShell";
import { Button } from "../components/ui";

export function LoginPage() {
  const { signIn, notice } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await signIn(username.trim(), password);
    } catch (err) {
      if (err instanceof ApiError && err.status === 429) setError("Too many attempts. Please wait a minute and try again.");
      else if (err instanceof ApiError && err.status === 401) setError("Incorrect username or password.");
      else if (err instanceof ApiError) setError(err.message);
      else setError("Could not reach the server. Check your connection and try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-full place-items-center bg-bg px-4 py-10">
      <div className="w-full max-w-sm space-y-6">
        <div className="flex justify-center">
          <Brand />
        </div>
        <form onSubmit={submit} className="space-y-4 rounded-2xl border border-line bg-panel p-6 shadow-card" aria-label="Sign in">
          <div>
            <h1 className="text-xl font-semibold">Sign in</h1>
            <p className="mt-1 text-sm text-muted">Private, offline answers from your own documents.</p>
          </div>
          {(notice || error) && (
            <div role="alert" className="rounded-lg border border-danger/30 bg-danger-soft px-3 py-2 text-sm text-danger">
              {error ?? notice}
            </div>
          )}
          <label className="block text-sm font-medium">
            Username
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
              className="mt-1 w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm"
            />
          </label>
          <label className="block text-sm font-medium">
            Password
            <div className="relative mt-1">
              <input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                type={show ? "text" : "password"}
                autoComplete="current-password"
                required
                className="w-full rounded-lg border border-line bg-bg px-3 py-2 pr-10 text-sm"
              />
              <button
                type="button"
                onClick={() => setShow((s) => !s)}
                aria-label={show ? "Hide password" : "Show password"}
                className="absolute inset-y-0 right-2 text-muted hover:text-ink"
              >
                {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
              </button>
            </div>
          </label>
          <Button type="submit" loading={busy} className="w-full">
            Sign in
          </Button>
        </form>
        <p className="flex items-center justify-center gap-1.5 text-xs text-muted">
          <Lock className="size-3" aria-hidden /> Your role comes from your account, never from this page.
        </p>
      </div>
    </div>
  );
}
