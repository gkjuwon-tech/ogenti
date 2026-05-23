import { prisma } from "@/lib/db";
import { getStripe, isStripeConfigured } from "@/server/stripe";

/**
 * Reports a single UsageEvent to Stripe as a metered usage record. Safe to
 * call repeatedly — uses the event's idempotency key on the Stripe side and
 * persists `reportedToStripeAt` locally.
 *
 * When Stripe is not configured (e.g. local development without keys), this
 * is a no-op that simply marks the row as reported so the UI can move on.
 */
export async function reportUsageEventToStripe(usageEventId: string) {
  const event = await prisma.usageEvent.findUnique({
    where: { id: usageEventId },
    include: { organization: true },
  });
  if (!event) {
    throw new Error(`UsageEvent ${usageEventId} not found.`);
  }
  if (event.reportedToStripeAt) return;

  const org = event.organization;

  if (!isStripeConfigured() || !org.stripeMeterItemId) {
    await prisma.usageEvent.update({
      where: { id: event.id },
      data: { reportedToStripeAt: new Date() },
    });
    return;
  }

  const stripe = getStripe();
  const record = await stripe.subscriptionItems.createUsageRecord(
    org.stripeMeterItemId,
    {
      quantity: event.billedSeconds,
      timestamp: Math.floor(event.createdAt.getTime() / 1000),
      action: "increment",
    },
    {
      idempotencyKey: `usage:${event.idempotencyKey}`,
    },
  );

  await prisma.usageEvent.update({
    where: { id: event.id },
    data: {
      reportedToStripeAt: new Date(),
      stripeUsageRecordId: record.id,
    },
  });
}

/**
 * Computes the rate (USD cents per generation second) the org currently pays.
 * Returns the overage rate for subscribers and the flat PAYG rate for
 * non-subscribers, both in cents.
 */
export async function currentRateCents(organizationId: string): Promise<number> {
  const org = await prisma.organization.findUnique({
    where: { id: organizationId },
  });
  if (!org) throw new Error(`Organization ${organizationId} not found.`);

  switch (org.planTier) {
    case "STARTER":
      return 120;
    case "STUDIO":
      return 95;
    case "AGENCY":
      return 75;
    case "ENTERPRISE":
      return 0; // billed offline
    case "PAYG":
    default:
      return 150;
  }
}

/**
 * The number of generation seconds included in the current billing period for
 * a given plan. Past this threshold each additional second is billed at the
 * overage / PAYG rate.
 */
export function planIncludedSeconds(planTier: string): number {
  switch (planTier) {
    case "STARTER":
      return 60;
    case "STUDIO":
      return 360;
    case "AGENCY":
      return 1800;
    case "ENTERPRISE":
      return Number.MAX_SAFE_INTEGER;
    case "PAYG":
    default:
      return 0;
  }
}
