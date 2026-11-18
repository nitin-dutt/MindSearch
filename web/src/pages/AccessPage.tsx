import { useEffect, useState } from "react";
import { Check, Crown, Layers, Users } from "lucide-react";
import { getPolicy } from "../api/client";
import type { Policy } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { Badge, Card, FieldLabel, Page, PageHeader, Skeleton } from "../components/ui";

export function AccessPage() {
  const { user } = useAuth();
  const [policy, setPolicy] = useState<Policy | null>(null);

  useEffect(() => {
    getPolicy().then(setPolicy).catch(() => setPolicy(null));
  }, []);

  if (!user) return null;
  return (
    <Page>
      <PageHeader
        title="My access"
        description="What your account can read and what you may share. The server enforces this on every search: this page only explains it."
      />

      <Card className="flex flex-wrap items-center gap-4">
        <div className="grid size-12 shrink-0 place-items-center rounded-full bg-[var(--warm)] text-lg font-semibold uppercase text-[var(--warm-ink)]" aria-hidden>
          {user.username.slice(0, 1)}
        </div>
        <div>
          <div className="font-semibold">{user.username}</div>
          <div className="mt-1 flex flex-wrap gap-1">
            <Badge tone="accent">
              <Layers className="size-3.5" aria-hidden /> level: {user.level}
            </Badge>
            {user.groups.length > 0 ? (
              user.groups.map((g) => (
                <Badge key={g}>
                  <Users className="size-3.5" aria-hidden /> {g}
                </Badge>
              ))
            ) : (
              <Badge>no department restriction</Badge>
            )}
            {user.is_super && (
              <Badge tone="warn">
                <Crown className="size-3.5" aria-hidden /> administrator
              </Badge>
            )}
          </div>
        </div>
      </Card>

      <Card className="space-y-4">
        <h2 className="text-base font-semibold">What you can read</h2>
        {user.is_super ? (
          <p className="text-sm text-muted">As an administrator you can read every document. That access is written to the audit log.</p>
        ) : (
          <ul className="space-y-3 text-sm text-muted">
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

      <Card className="space-y-4">
        <h2 className="text-base font-semibold">What you can share when you upload</h2>
        {!policy ? (
          <Skeleton className="h-16" />
        ) : (
          <div className="space-y-6 text-sm">
            <div>
              <FieldLabel>Access levels you may set</FieldLabel>
              <div className="flex flex-wrap gap-1">
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
                <FieldLabel>Departments you may choose</FieldLabel>
                <div className="flex flex-wrap gap-1">
                  {policy.groups.map((g) => (
                    <Badge key={g}>{g}</Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </Card>

      <Card className="space-y-4">
        <h2 className="text-base font-semibold">Deleting documents</h2>
        {user.is_super && <p className="text-sm text-muted">As an administrator you can delete any document. Every delete is written to the audit log.</p>}
        {policy ? <p className="text-sm text-muted">{policy.delete_rule.text}</p> : <Skeleton className="h-10" />}
      </Card>
    </Page>
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
