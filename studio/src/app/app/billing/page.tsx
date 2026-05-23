import Link from "next/link";
import { prisma } from "@/lib/db";
import { requireOrgContext } from "@/server/orgs";
import { getCurrentCycleSnapshot } from "@/server/billing/cycle";
import {
  formatDateRange,
  formatSeconds,
  formatUsdCents,
} from "@/lib/format";
import { PLANS, PAYG_RATE_USD, formatUsd } from "@/lib/pricing";
import { BillingActions } from "./BillingActions";
import styles from "./billing.module.css";

export const dynamic = "force-dynamic";

export default async function BillingPage() {
  const ctx = await requireOrgContext();
  const [org, invoices, cycle] = await Promise.all([
    prisma.organization.findUnique({ where: { id: ctx.organizationId } }),
    prisma.invoice.findMany({
      where: { organizationId: ctx.organizationId },
      orderBy: { createdAt: "desc" },
      take: 24,
    }),
    getCurrentCycleSnapshot(ctx.organizationId),
  ]);

  if (!org) return null;

  const usagePct = cycle.includedSeconds
    ? Math.min(100, (cycle.usedSeconds / cycle.includedSeconds) * 100)
    : 0;

  return (
    <>
      <header className={styles.header}>
        <div>
          <p className="eyebrow">Billing</p>
          <h1 className={styles.title}>Plan, usage, and invoices</h1>
          <p className={styles.subtitle}>
            Hybrid subscription + metered model. Subscriptions cover an
            included pool of generation seconds; anything beyond is billed at
            the overage rate or your pay-as-you-go rate.
          </p>
        </div>
      </header>

      {org.generationPaused && (
        <section className={styles.banner} data-tone="warning">
          <strong>Generation paused.</strong> A recent invoice did not collect
          successfully. Update your payment method to resume rendering.
        </section>
      )}

      <section className={styles.cycle}>
        <header className={styles.cycleHead}>
          <div>
            <p className="eyebrow">Current billing period</p>
            <h2 className={styles.cycleRange}>
              {formatDateRange(cycle.cycleStart, cycle.cycleEnd)}
            </h2>
          </div>
          <BillingActions
            planTier={org.planTier}
            hasStripeCustomer={Boolean(org.stripeCustomerId)}
          />
        </header>

        <div className={styles.cycleGrid}>
          <article className={styles.cycleMetric}>
            <h3 className={styles.metricLabel}>Used this cycle</h3>
            <p className={`mono ${styles.metricValue}`}>
              {formatSeconds(cycle.usedSeconds)}
            </p>
            <p className={styles.metricSub}>
              of {cycle.includedSeconds > 0 ? formatSeconds(cycle.includedSeconds) : "no"} included
            </p>
            {cycle.includedSeconds > 0 && (
              <div className={styles.bar} aria-hidden>
                <span style={{ width: `${usagePct}%` }} />
              </div>
            )}
          </article>
          <article className={styles.cycleMetric}>
            <h3 className={styles.metricLabel}>Base subscription</h3>
            <p className={`mono ${styles.metricValue}`}>
              {formatUsdCents(cycle.baseCostCents)}
            </p>
            <p className={styles.metricSub}>
              {prettyPlan(cycle.planTier)} plan
            </p>
          </article>
          <article className={styles.cycleMetric}>
            <h3 className={styles.metricLabel}>Metered usage</h3>
            <p className={`mono ${styles.metricValue}`}>
              {formatUsdCents(cycle.overageCostCents)}
            </p>
            <p className={styles.metricSub}>
              {formatSeconds(cycle.overageSeconds)} billed at overage
            </p>
          </article>
          <article className={styles.cycleMetric}>
            <h3 className={styles.metricLabel}>Projected charge</h3>
            <p className={`mono ${styles.metricValue}`}>
              {formatUsdCents(cycle.totalCostCents)}
            </p>
            <p className={styles.metricSub}>Net-30 on Agency and above</p>
          </article>
        </div>
      </section>

      <section className={styles.plans}>
        <header className={styles.plansHead}>
          <h2 className={styles.sectionTitle}>Switch plan</h2>
          <p className={styles.sectionSub}>
            Subscription changes take effect at the next billing cycle.
            Pay-as-you-go can be turned on alongside any subscription tier.
          </p>
        </header>

        <ul className={styles.planList}>
          {PLANS.filter((p) => p.id !== "enterprise").map((p) => {
            const current =
              org.planTier === p.id.toUpperCase() ||
              (p.id === "starter" && org.planTier === "PAYG" && false); // never auto-mark PAYG as Starter
            return (
              <li
                key={p.id}
                className={styles.planRow}
                data-current={current ? "true" : undefined}
              >
                <div className={styles.planMain}>
                  <h3 className={styles.planName}>{p.name}</h3>
                  <p className={styles.planTagline}>{p.tagline}</p>
                </div>
                <div className={styles.planNumbers}>
                  <span className={`mono ${styles.planPrice}`}>
                    {p.monthlyUsd != null
                      ? formatUsd(p.monthlyUsd) + " / mo"
                      : "Custom"}
                  </span>
                  <span className={styles.planIncl}>
                    {p.includedSeconds.toLocaleString()} s included
                  </span>
                </div>
                <BillingActions
                  planTier={org.planTier}
                  hasStripeCustomer={Boolean(org.stripeCustomerId)}
                  switchTo={p.id.toUpperCase() as "STARTER" | "STUDIO" | "AGENCY"}
                  isCurrent={current}
                />
              </li>
            );
          })}
          <li
            className={styles.planRow}
            data-current={org.planTier === "PAYG" ? "true" : undefined}
          >
            <div className={styles.planMain}>
              <h3 className={styles.planName}>Pay-as-you-go</h3>
              <p className={styles.planTagline}>
                No commitment, billed in arrears at the end of each calendar
                month.
              </p>
            </div>
            <div className={styles.planNumbers}>
              <span className={`mono ${styles.planPrice}`}>
                {formatUsd(PAYG_RATE_USD, { fractional: true })} / s
              </span>
              <span className={styles.planIncl}>No included pool</span>
            </div>
            <BillingActions
              planTier={org.planTier}
              hasStripeCustomer={Boolean(org.stripeCustomerId)}
              switchTo="PAYG"
              isCurrent={org.planTier === "PAYG"}
            />
          </li>
        </ul>
      </section>

      <section className={styles.invoices}>
        <header className={styles.invoicesHead}>
          <h2 className={styles.sectionTitle}>Invoices</h2>
          <p className={styles.sectionSub}>
            Historical invoices synced from Stripe via the billing webhook.
          </p>
        </header>
        {invoices.length === 0 ? (
          <p className={styles.empty}>
            No invoices yet. Your first invoice will appear at the close of the
            current billing period.
          </p>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Number</th>
                <th>Period</th>
                <th>Status</th>
                <th>Amount</th>
                <th aria-label="actions" />
              </tr>
            </thead>
            <tbody>
              {invoices.map((inv) => (
                <tr key={inv.id}>
                  <td className="mono">{inv.number ?? inv.stripeInvoiceId.slice(0, 12)}</td>
                  <td className="mono">
                    {formatDateRange(inv.periodStart, inv.periodEnd)}
                  </td>
                  <td>
                    <span className={styles.invStatus} data-status={inv.status}>
                      {inv.status.toLowerCase()}
                    </span>
                  </td>
                  <td className="mono">{formatUsdCents(inv.amountDueCents)}</td>
                  <td className={styles.actions}>
                    {inv.hostedInvoiceUrl ? (
                      <Link
                        href={inv.hostedInvoiceUrl}
                        prefetch={false}
                        className={styles.actionLink}
                      >
                        View →
                      </Link>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
