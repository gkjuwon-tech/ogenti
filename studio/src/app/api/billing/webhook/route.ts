import { NextResponse } from "next/server";
import type Stripe from "stripe";
import { prisma } from "@/lib/db";
import { getStripe } from "@/server/stripe";

export const runtime = "nodejs";

/**
 * Stripe webhook handler. Validates the signature, then dispatches on the
 * event type. Designed to be idempotent — every persisted side-effect is
 * guarded by an upsert or status check.
 *
 * Configure with `stripe listen --forward-to localhost:3000/api/billing/webhook`
 * in development and a Stripe-issued webhook URL in production.
 */
export async function POST(req: Request) {
  const secret = process.env.STRIPE_WEBHOOK_SECRET;
  const signature = req.headers.get("stripe-signature");
  if (!secret || !signature) {
    return NextResponse.json(
      { error: "webhook_misconfigured" },
      { status: 500 },
    );
  }

  const body = await req.text();
  let event: Stripe.Event;
  try {
    event = getStripe().webhooks.constructEvent(body, signature, secret);
  } catch (err) {
    console.error("webhook signature verification failed", err);
    return NextResponse.json(
      { error: "invalid_signature" },
      { status: 400 },
    );
  }

  try {
    switch (event.type) {
      case "customer.subscription.created":
      case "customer.subscription.updated": {
        const sub = event.data.object as Stripe.Subscription;
        await syncSubscription(sub);
        break;
      }
      case "customer.subscription.deleted": {
        const sub = event.data.object as Stripe.Subscription;
        await prisma.organization.updateMany({
          where: { stripeSubscriptionId: sub.id },
          data: {
            subscriptionStatus: "CANCELED",
            generationPaused: true,
          },
        });
        break;
      }
      case "invoice.paid": {
        const inv = event.data.object as Stripe.Invoice;
        await upsertInvoice(inv);
        if (inv.customer) {
          await prisma.organization.updateMany({
            where: { stripeCustomerId: String(inv.customer) },
            data: { generationPaused: false },
          });
        }
        break;
      }
      case "invoice.payment_failed": {
        const inv = event.data.object as Stripe.Invoice;
        await upsertInvoice(inv);
        if (inv.customer) {
          await prisma.organization.updateMany({
            where: { stripeCustomerId: String(inv.customer) },
            data: { generationPaused: true, subscriptionStatus: "PAST_DUE" },
          });
        }
        break;
      }
      case "invoice.finalized":
      case "invoice.updated": {
        const inv = event.data.object as Stripe.Invoice;
        await upsertInvoice(inv);
        break;
      }
      default:
        // Other events are intentionally ignored. Add cases as the product grows.
        break;
    }
  } catch (err) {
    console.error("webhook handler failed", err, event.id);
    return NextResponse.json(
      { error: "handler_failed", eventId: event.id },
      { status: 500 },
    );
  }

  return NextResponse.json({ received: true, type: event.type });
}

async function syncSubscription(sub: Stripe.Subscription) {
  const customerId = typeof sub.customer === "string" ? sub.customer : sub.customer.id;
  const planFromMeta = sub.metadata?.ogenti_plan as
    | "STARTER"
    | "STUDIO"
    | "AGENCY"
    | "ENTERPRISE"
    | "PAYG"
    | undefined;
  const meterItem = sub.items.data.find((it) => it.price.recurring?.usage_type === "metered");

  await prisma.organization.updateMany({
    where: { stripeCustomerId: customerId },
    data: {
      stripeSubscriptionId: sub.id,
      stripeMeterItemId: meterItem?.id ?? null,
      subscriptionStatus: mapStatus(sub.status),
      planTier: planFromMeta ?? undefined,
      currentPeriodStart: new Date(sub.current_period_start * 1000),
      currentPeriodEnd: new Date(sub.current_period_end * 1000),
      generationPaused:
        sub.status === "past_due" ||
        sub.status === "unpaid" ||
        sub.status === "canceled",
    },
  });
}

async function upsertInvoice(inv: Stripe.Invoice) {
  const customerId =
    typeof inv.customer === "string" ? inv.customer : inv.customer?.id ?? null;
  if (!customerId) return;
  const org = await prisma.organization.findFirst({
    where: { stripeCustomerId: customerId },
  });
  if (!org) return;

  await prisma.invoice.upsert({
    where: { stripeInvoiceId: inv.id },
    create: {
      organizationId: org.id,
      stripeInvoiceId: inv.id,
      number: inv.number ?? null,
      status: mapInvoiceStatus(inv.status),
      amountDueCents: inv.amount_due,
      amountPaidCents: inv.amount_paid,
      currency: inv.currency,
      hostedInvoiceUrl: inv.hosted_invoice_url ?? null,
      pdfUrl: inv.invoice_pdf ?? null,
      periodStart: new Date(inv.period_start * 1000),
      periodEnd: new Date(inv.period_end * 1000),
      paidAt: inv.status === "paid" ? new Date() : null,
    },
    update: {
      number: inv.number ?? null,
      status: mapInvoiceStatus(inv.status),
      amountDueCents: inv.amount_due,
      amountPaidCents: inv.amount_paid,
      currency: inv.currency,
      hostedInvoiceUrl: inv.hosted_invoice_url ?? null,
      pdfUrl: inv.invoice_pdf ?? null,
      paidAt: inv.status === "paid" ? new Date() : null,
    },
  });
}

function mapStatus(s: Stripe.Subscription.Status): string {
  switch (s) {
    case "trialing":
      return "TRIALING";
    case "active":
      return "ACTIVE";
    case "past_due":
      return "PAST_DUE";
    case "canceled":
      return "CANCELED";
    case "incomplete":
    case "incomplete_expired":
      return "INCOMPLETE";
    case "paused":
      return "PAUSED";
    default:
      return "INCOMPLETE";
  }
}

function mapInvoiceStatus(s: Stripe.Invoice.Status | null): string {
  switch (s) {
    case "paid":
      return "PAID";
    case "open":
      return "OPEN";
    case "uncollectible":
      return "UNCOLLECTIBLE";
    case "void":
      return "VOID";
    case "draft":
    default:
      return "DRAFT";
  }
}
