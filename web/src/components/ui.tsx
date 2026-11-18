import { useEffect, useRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { Loader2 } from "lucide-react";

export function Button({
  variant = "primary",
  loading = false,
  className = "",
  children,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; loading?: boolean }) {
  return (
    <button {...rest} disabled={disabled || loading} className={`${buttonClass(variant)} ${className}`}>
      {loading && <Loader2 className="size-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

type ButtonVariant = "primary" | "secondary" | "danger" | "ghost";

/** The one button look, also used by links that act as buttons (for example "Upload" on the Documents page), so every primary action matches.
 *  Height 36 (h-9), radius 8, 16px icons. A press scales down a touch; hover is a colour step only. */
export function buttonClass(variant: ButtonVariant = "primary"): string {
  const styles = {
    primary: "bg-brand hover:brightness-95",
    secondary: "border border-line bg-panel text-ink hover:bg-accent-soft",
    danger: "bg-danger text-white hover:opacity-90",
    ghost: "text-muted hover:bg-accent-soft hover:text-ink",
  }[variant];
  return `inline-flex h-9 items-center justify-center gap-2 rounded-lg px-4 text-sm font-medium transition-[background-color,border-color,color,filter,opacity,transform] duration-150 ease-out active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50 disabled:active:scale-100 ${styles}`;
}

export function Badge({ tone = "neutral", children }: { tone?: "neutral" | "ok" | "warn" | "danger" | "accent"; children: ReactNode }) {
  const styles = {
    neutral: "bg-bg text-muted border-line",
    ok: "bg-ok-soft text-ok border-transparent",
    warn: "bg-warn-soft text-warn border-transparent",
    danger: "bg-danger-soft text-danger border-transparent",
    accent: "bg-accent-soft text-accent border-transparent",
  }[tone];
  return <span className={`inline-flex h-6 items-center gap-1 rounded-md border px-2 text-xs font-medium ${styles}`}>{children}</span>;
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-md bg-line/70 ${className}`} aria-hidden />;
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-line px-6 py-14 text-center">
      <div className="rounded-xl bg-brand p-3">{icon}</div>
      <h3 className="text-base font-semibold">{title}</h3>
      {children && <div className="max-w-md text-sm text-muted">{children}</div>}
    </div>
  );
}

export function Card({ className = "", children }: { className?: string; children: ReactNode }) {
  return <section className={`rounded-2xl border border-line bg-panel p-6 shadow-card ${className}`}>{children}</section>;
}

/** Every page except Chat sits in this column: one width, one gutter and one top offset, so titles start at the same place on every page. */
export function Page({ className = "", children }: { className?: string; children: ReactNode }) {
  return <div className={`mx-auto w-full max-w-4xl space-y-6 px-4 py-6 md:px-8 md:py-8 ${className}`}>{children}</div>;
}

/** The page title, a one-line explanation under it, and at most one action on the right, centred on the title block. */
export function PageHeader({ title, description, action }: { title: string; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-lg font-semibold">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-sm text-muted">{description}</p>}
      </div>
      {action}
    </div>
  );
}

/** A small label above a group of chips or fields: sentence case, never uppercase. */
export function FieldLabel({ children }: { children: ReactNode }) {
  return <div className="mb-2 text-xs font-medium text-muted">{children}</div>;
}

/** A keyboard key, for shortcut hints such as Enter. */
export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="inline-grid h-5 min-w-5 place-items-center rounded border border-line bg-panel px-1 font-mono text-[11px] leading-none text-muted">{children}</kbd>;
}

/** A modal confirmation built on the native <dialog>: focus is trapped, Escape closes it, and it is announced to screen readers. */
export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel,
  onConfirm,
  onCancel,
  busy = false,
}: {
  open: boolean;
  title: string;
  children: ReactNode;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
  busy?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal?.();
    if (!open && dialog.open) dialog.close?.();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onCancel={(e) => {
        e.preventDefault();
        onCancel();
      }}
      aria-labelledby="confirm-title"
      className="m-auto w-[min(26rem,calc(100vw-2rem))] rounded-2xl border border-line bg-panel p-0 text-ink shadow-2xl backdrop:bg-black/50"
    >
      {open && (
        <div className="space-y-4 p-6">
          <h2 id="confirm-title" className="text-base font-semibold">
            {title}
          </h2>
          <div className="text-sm text-muted">{children}</div>
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={onCancel} disabled={busy}>
              Cancel
            </Button>
            <Button variant="danger" onClick={onConfirm} loading={busy}>
              {confirmLabel}
            </Button>
          </div>
        </div>
      )}
    </dialog>
  );
}
