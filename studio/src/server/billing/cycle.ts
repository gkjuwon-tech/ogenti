import { prisma } from "@/lib/db";
import { planIncludedSeconds } from "@/server/billing/usage";

export interface CycleSnapshot {
  organizationId: string;
  planTier: string;
  cycleStart: Date;
  cycleEnd: Date;
  includedSeconds: number;
  usedSeconds: number;
  overageSeconds: number;
  overageCostCents: number;
  baseCostCents: number;
  totalCostCents: number;
  generationCount: number;
}

/**
 * Computes a current-period billing snapshot for an organization. Used by
 * the dashboard, the billing page, and the prepaid-warning banner.
 *
 * Defaults to a calendar-month window if no `currentPeriodStart` /
 * `currentPeriodEnd` is mirrored from Stripe yet (e.g. on a brand-new tenant
 * before the first invoice cycle).
 */
export async function getCurrentCycleSnapshot(
  organizationId: string,
): Promise<CycleSnapshot> {
  const org = await prisma.organization.findUnique({
    where: { id: organizationId },
  });
  if (!org) throw new Error(`Organization ${organizationId} not found.`);

  const now = new Date();
  const cycleStart =
    org.currentPeriodStart ??
    new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
  const cycleEnd =
    org.currentPeriodEnd ??
    new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1));

  const usageEvents = await prisma.usageEvent.findMany({
    where: {
      organizationId,
      createdAt: { gte: cycleStart, lt: cycleEnd },
    },
    select: { billedSeconds: true, costCents: true },
  });

  const generationCount = await prisma.generation.count({
    where: {
      organizationId,
      createdAt: { gte: cycleStart, lt: cycleEnd },
      status: "SUCCEEDED",
    },
  });

  const usedSeconds = usageEvents.reduce((a, e) => a + e.billedSeconds, 0);
  const realizedCostCents = usageEvents.reduce((a, e) => a + e.costCents, 0);

  const included = planIncludedSeconds(org.planTier);
  const overageSeconds = Math.max(0, usedSeconds - included);

  // Reconstruct billed cost from the per-event captured rate (overageRate is
  // always already encoded into each event's costCents at write-time).
  return {
    organizationId,
    planTier: org.planTier,
    cycleStart,
    cycleEnd,
    includedSeconds: included,
    usedSeconds,
    overageSeconds,
    overageCostCents: realizedCostCents,
    baseCostCents: planBaseCostCents(org.planTier),
    totalCostCents: planBaseCostCents(org.planTier) + realizedCostCents,
    generationCount,
  };
}

function planBaseCostCents(tier: string): number {
  switch (tier) {
    case "STARTER":
      return 9900;
    case "STUDIO":
      return 49900;
    case "AGENCY":
      return 199900;
    case "ENTERPRISE":
      return 0;
    case "PAYG":
    default:
      return 0;
  }
}
