import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { FileText, LogOut, Menu, MessageSquare, Moon, ChevronsLeft, ChevronsRight, ShieldCheck, Sun, Upload, X } from "lucide-react";
import { useAuth } from "../auth/AuthContext";
import { Badge } from "./ui";
import { MindMark } from "./MindMark";

const NAV = [
  { to: "/", label: "Chat", icon: MessageSquare, end: true },
  { to: "/documents", label: "Documents", icon: FileText, end: false },
  { to: "/upload", label: "Upload", icon: Upload, end: false },
  { to: "/access", label: "My access", icon: ShieldCheck, end: false },
];

type Theme = "light" | "dark" | "system";

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem("mindsearch.theme") as Theme) || "system";
    } catch {
      return "system";
    }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("mindsearch.theme", theme);
    } catch {
      /* a remembered theme is a convenience only */
    }
  }, [theme]);
  const cycle = () => setTheme((t) => (t === "system" ? "light" : t === "light" ? "dark" : "system"));
  return [theme, cycle];
}

function useCollapsed(): [boolean, () => void] {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem("mindsearch.sidebar") === "collapsed";
    } catch {
      return false;
    }
  });
  const toggle = () =>
    setCollapsed((c) => {
      try {
        localStorage.setItem("mindsearch.sidebar", c ? "open" : "collapsed");
      } catch {
        /* a remembered sidebar state is a convenience only */
      }
      return !c;
    });
  return [collapsed, toggle];
}

export function AppShell() {
  const { user, signOut, devUsers, switchTo } = useAuth();
  const [theme, cycleTheme] = useTheme();
  const [open, setOpen] = useState(false);
  const [collapsed, toggleCollapsed] = useCollapsed();
  const ThemeIcon = theme === "dark" ? Moon : Sun;

  // `slim` = the icon-only sidebar. The phone menu is never slim.
  const renderNav = (slim: boolean) => (
    <nav aria-label="Main" className="flex flex-1 flex-col gap-1">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          title={slim ? label : undefined}
          onClick={() => setOpen(false)}
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-lg py-2 text-sm font-medium transition ${slim ? "justify-center px-0" : "px-3"} ${
              isActive ? "nav-active text-accent" : "text-muted hover:bg-accent-soft hover:text-ink"
            }`
          }
        >
          <Icon className={slim ? "size-5" : "size-4"} aria-hidden />
          <span className={slim ? "sr-only" : ""}>{label}</span>
        </NavLink>
      ))}
    </nav>
  );

  const renderAccount = (slim: boolean) => (
    <div className="space-y-3 border-t border-line pt-4">
      <div className={`flex items-center gap-3 ${slim ? "justify-center" : ""}`} title={slim ? `${user?.username} (${user?.level})` : undefined}>
        <div className="grid size-9 shrink-0 place-items-center rounded-full bg-[var(--warm)] text-sm font-semibold uppercase text-[var(--warm-ink)]" aria-hidden>
          {user?.username.slice(0, 1)}
        </div>
        {!slim && (
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold">{user?.username}</div>
            <div className="mt-0.5 flex flex-wrap gap-1">
              <Badge tone="accent">{user?.level}</Badge>
              {user?.is_super && <Badge tone="warn">admin</Badge>}
            </div>
          </div>
        )}
      </div>
      {!slim && import.meta.env.DEV && devUsers.length > 0 && (
        <label className="block rounded-lg border border-dashed border-line px-3 py-2 text-xs text-muted">
          Dev: switch account
          <select
            value={user?.username ?? ""}
            onChange={(e) => void switchTo(e.target.value)}
            className="mt-1 w-full rounded-md border border-line bg-bg px-2 py-1 text-xs text-ink"
          >
            {devUsers.map((u) => (
              <option key={u.username} value={u.username}>
                {u.label} ({u.username})
              </option>
            ))}
          </select>
        </label>
      )}
      <div className={`flex gap-2 ${slim ? "flex-col" : ""}`}>
        <button
          onClick={cycleTheme}
          title={slim ? `Theme: ${theme}` : undefined}
          className="flex flex-1 items-center justify-center gap-2 rounded-lg border border-line px-3 py-1.5 text-xs text-muted hover:text-ink"
          aria-label={`Theme: ${theme}. Click to change`}
        >
          <ThemeIcon className="size-3.5" aria-hidden /> {!slim && theme}
        </button>
        <button
          onClick={signOut}
          title={slim ? "Sign out" : undefined}
          aria-label="Sign out"
          className="flex flex-1 items-center justify-center gap-2 rounded-lg border border-line px-3 py-1.5 text-xs text-muted hover:text-danger"
        >
          <LogOut className="size-3.5" aria-hidden /> {!slim && "Sign out"}
        </button>
      </div>
    </div>
  );

  return (
    <div className="flex h-full">
      <aside
        className={`sidebar hidden shrink-0 flex-col gap-6 border-r border-line transition-[width] duration-200 md:flex ${collapsed ? "w-[4.75rem] p-3" : "w-64 p-5"}`}
      >
        <div className={`flex items-center ${collapsed ? "flex-col gap-4" : "justify-between"}`}>
          <Brand compact={collapsed} />
          <button
            type="button"
            onClick={toggleCollapsed}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-expanded={!collapsed}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="rounded p-1 text-muted transition hover:text-ink"
          >
            {collapsed ? <ChevronsRight className="size-4" aria-hidden /> : <ChevronsLeft className="size-4" aria-hidden />}
          </button>
        </div>
        {renderNav(collapsed)}
        {renderAccount(collapsed)}
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sidebar flex items-center justify-between border-b border-line px-4 py-3 md:hidden">
          <Brand />
          <button aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open} onClick={() => setOpen((o) => !o)} className="rounded-lg p-2 text-muted hover:bg-accent-soft">
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </header>
        {open && (
          <div className="sidebar flex flex-col gap-6 border-b border-line p-4 md:hidden">
            {renderNav(false)}
            {renderAccount(false)}
          </div>
        )}
        <main className="min-h-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

/** The logo is just the mind symbol and the name. The symbol is the target of the "document goes into the mind" upload effect. */
export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="grid size-9 place-items-center rounded-full" data-mind-target>
        <MindMark className="size-8" />
      </span>
      {!compact && <span className="text-base font-semibold tracking-tight">MindSearch</span>}
    </div>
  );
}
