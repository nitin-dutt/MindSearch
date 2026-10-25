import { useEffect, useState } from "react";
import { Check, Crown, Layers, ShieldCheck, Users } from "lucide-react";
import { getPolicy } from "../api/client";
import type { Policy } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { Badge, Card, Skeleton } from "../components/ui";

export function AccessPage() {
  const { user } = useAuth();
  const [policy, setPolicy] = useState<Policy | null>(null);

  useEffect(() => {
    getPolicy().then(setPolicy).catch(() => setPolicy(null));
  }, []);

  if (!user) return null;
  return (
    <div className="mx-auto max-w-3xl space-y-5 p-4 md:p-8">
      <div>
        <h1 className="text-lg font-semibold">My access</h1>
        <p className="text-sm text-muted">What your account can read and what you may share. The server enforces this on every search: this page only explains it.</p>
      </div>

      <Card className="flex flex-wrap items-center gap-4">
        <div className="grid size-12 place-items-center rounded-full bg-accent text-lg font-semibold uppercase text-accent-ink" aria-hidden>
          {user.username.slice(0, 1)}
        </div>
        <div>
          <div className="font-semibold">{user.username}</div>
          <div className="mt-1 flex flex-wrap gap-1.5">
            <Badge tone="accent">
              <Layers className="size-3" aria-hidden /> level: {user.level}
            </Badge>
            {user.groups.length > 0 ? (
              user.groups.map((g) => (
                <Badge key={g}>
                  <Users className="size-3" aria-hidden /> {g}
                </Badge>
              ))
            ) : (
              <Badge>no department restriction</Badge>
            )}
            {user.is_super && (
              <Badge tone="warn">
                <Crown className="size-3" aria-hidden /> administrator
              </Badge>
            )}
          </div>
        </div>
      </Card>

      <Card className="space-y-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <ShieldCheck className="size-4 text-accent" aria-hidden /> What you can read
        </h2>
        {user.is_super ? (
          <p className="text-sm text-muted">As an administrator you can read every document. That access is written to the audit log.</p>
        ) : (
          <ul className="space-y-2 text-sm text-muted">
            <Rule>
              Documents at level <strong className="text-ink">{user.level}</strong> or lower.
            </Rule>
            <Rule>
              {user.groups.length > 0 ? (
                <>
                  Documents for everyone, or limited to your departments: <strong className="text-ink">{user.groups.join(", ")}</strong>.
                </>
              ) : (
                <>Documents that are not limited to a department.</>
              )}
            </Rule>
            <Rule>Anything else is invisible to you: it is never searched, quoted or counted, and asking about it returns “I couldn’t find any relevant information”.</Rule>
          </ul>
        )}
      </Card>

      <Card className="space-y-3">
        <h2 className="text-sm font-semibold">What you can share when you upload</h2>
        {!policy ? (
          <Skeleton className="h-16" />
        ) : (
          <div className="space-y-3 text-sm">
            <div>
              <div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted">Access levels you may set</div>
              <div className="flex flex-wrap gap-1.5">
                {policy.levels.map((l) => (
                  <Badge key={l} tone={l === user.level ? "accent" : "neutral"}>
                    {l}
                  </Badge>
                ))}
              </div>
              <p className="mt-2 text-muted">You can never grant more access than you have yourself.</p>
            </div>
            {policy.groups.length > 0 && (
              <div>
                <div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted">Departments you may choose</div>
                <div className="flex flex-wrap gap-1.5">
                  {policy.groups.map((g) => (
                    <Badge key={g}>{g}</Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </Card>

      <Card className="space-y-2">
        <h2 className="text-sm font-semibold">Deleting documents</h2>
        <p className="text-sm text-muted">{user.is_super ? "As an administrator you can delete any document." : "You can delete the documents you uploaded and those uploaded by people at a lower level than yours. Only an administrator can delete other documents."}</p>
      </Card>
    </div>
  );
}

function Rule({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2">
      <Check className="mt-0.5 size-4 shrink-0 text-ok" aria-hidden />
      <span>{children}</span>
    </li>
  );
}
