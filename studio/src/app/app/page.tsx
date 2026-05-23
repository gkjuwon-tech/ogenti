import Link from "next/link";
import { prisma } from "@/lib/db";
import { requireOrgContext } from "@/server/orgs";
import { getCurrentCycleSnapshot } from "@/server/billing/cycle";
import { formatRelative, formatSeconds, formatUsdCents } from "@/lib/format";
import { ButtonLink } from "@/components/ui/Button";
import styles from "./page.module.css";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const ctx = await requireOrgContext();
  const cycle = await getCurrentCycleSnapshot(ctx.organizationId);
  const recent = await prisma.generation.findMany({
    where: { organizationId: ctx.organizationId },
    orderBy: { createdAt: "desc" },
    take: 6,
  });

  const usagePct = cycle.includedSeconds
    ? Math.min(100, (cycle.usedSeconds / cycle.includedSeconds) * 100)
    : 0;

  return (
    <>
      <header className={styles.header}>
        <div>
          <p className="eyebrow">Workspace overview</p>
          <h1 className={styles.title}>{ctx.organizationName}</h1>
          <p className={styles.subtitle}>
            Signed in as <span className="mono">{ctx.email}</span>
          </p>
        </div>
        <ButtonLink href="/app/generate" variant="primary" size="md">
          New generation
        </ButtonLink>
      </header>

      <section className={styles.grid}>
        <article className={styles.metric}>
          <h2 className={styles.metricLabel}>This cycle, used</h2>
          <p className={`mono ${styles.metricValue}`}>
            {formatSeconds(cycle.usedSeconds)}
          </p>
          <p className={styles.metricSub}>
            of {cycle.includedSeconds > 0 ? formatSeconds(cycle.includedSeconds) : "no included"} included
          </p>
          {cycle.includedSeconds > 0 && (
            <div className={styles.bar} aria-hidden>
              <span style={{ width: `${usagePct}%` }} />
            </div>
          )}
        </article>

        <article className={styles.metric}>
          <h2 className={styles.metricLabel}>Projected charge</h2>
          <p className={`mono ${styles.metricValue}`}>
            {formatUsdCents(cycle.totalCostCents)}
          </p>
          <p className={styles.metricSub}>
            {cycle.overageSeconds > 0
              ? `${formatSeconds(cycle.overageSeconds)} over plan included`
              : "Within plan allowance"}
          </p>
        </article>

        <article className={styles.metric}>
          <h2 className={styles.metricLabel}>Generations this cycle</h2>
          <p className={`mono ${styles.metricValue}`}>{cycle.generationCount}</p>
          <p className={styles.metricSub}>Successful renders only</p>
        </article>

        <article className={styles.metric}>
          <h2 className={styles.metricLabel}>Active plan</h2>
          <p className={`mono ${styles.metricValue}`}>{prettyPlan(cycle.planTier)}</p>
          <p className={styles.metricSub}>
            <Link href="/app/billing" className={styles.metricLink}>
              Manage billing →
            </Link>
          </p>
        </article>
      </section>

      <section className={styles.section}>
        <header className={styles.sectionHead}>
          <h2 className={styles.sectionTitle}>Recent generations</h2>
          <Link href="/app/library" className={styles.viewAll}>
            View library →
          </Link>
        </header>

        {recent.length === 0 ? (
          <p className={styles.empty}>
            Nothing here yet. Start your first generation to see results appear
            in this list.
          </p>
        ) : (
          <ul className={styles.recentList}>
            {recent.map((g) => (
              <li key={g.id} className={styles.recentRow}>
                <div className={styles.recentMain}>
                  <p className={styles.recentPrompt}>
                    {g.prompt.length > 120
                      ? g.prompt.slice(0, 120) + "…"
                      : g.prompt}
                  </p>
                  <p className={styles.recentMeta}>
                    <span className="mono">{g.resolution}</span>
                    <span aria-hidden> · </span>
                    <span className="mono">{g.durationSeconds}s</span>
                    <span aria-hidden> · </span>
                    <span>{formatRelative(g.createdAt)}</span>
                  </p>
                </div>
                <span className={styles.status} data-status={g.status}>
                  {prettyStatus(g.status)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

function prettyPlan(tier: string): string {
  switch (tier) {
    case "STARTER":
      return "Starter";
    case "STUDIO":
      return "Studio";
    case "AGENCY":
      return "Agency";
    case "ENTERPRISE":
      return "Enterprise";
    case "PAYG":
    default:
      return "Pay-as-you-go";
  }
}

function prettyStatus(s: string): string {
  switch (s) {
    case "QUEUED":
      return "Queued";
    case "RUNNING":
      return "Rendering";
    case "SUCCEEDED":
      return "Ready";
    case "FAILED":
      return "Failed";
    case "CANCELED":
      return "Canceled";
    default:
      return s;
  }
}
