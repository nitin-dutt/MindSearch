// Local-development shortcut: skip the login screen and switch between demo accounts from a dropdown.
//
// It is OFF unless ALL of these hold: the app runs under `npm run dev` (import.meta.env.DEV is false in a production build, so the code
// path is removed from the bundle), VITE_DEV_AUTOLOGIN=1, and the accounts are listed in VITE_DEV_USERS. Those values live in the
// git-ignored `.env.development.local`; no password is ever written in the source, and the Docker image does not copy `.env*` files.

export interface DevUser {
  label: string;
  username: string;
  password: string;
}

/** The demo accounts from the environment, or [] when the shortcut is not enabled or the value is malformed. */
export function parseDevUsers(enabled: boolean, flag: unknown, raw: unknown): DevUser[] {
  if (!enabled || flag !== "1" || typeof raw !== "string") return [];
  try {
    const list: unknown = JSON.parse(raw);
    if (!Array.isArray(list)) return [];
    return list.filter(
      (u): u is DevUser => !!u && typeof u.username === "string" && typeof u.password === "string" && typeof u.label === "string",
    );
  } catch {
    return [];
  }
}

export const DEV_USERS: DevUser[] = parseDevUsers(import.meta.env.DEV, import.meta.env.VITE_DEV_AUTOLOGIN, import.meta.env.VITE_DEV_USERS);

/** The account to sign in as at start-up: VITE_DEV_DEFAULT if it is listed, else the first one. */
export function defaultDevUser(users: DevUser[] = DEV_USERS, wanted: unknown = import.meta.env.VITE_DEV_DEFAULT): DevUser | null {
  return users.find((u) => u.username === wanted) ?? users[0] ?? null;
}
