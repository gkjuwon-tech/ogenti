import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import {
  requireOrgContext,
  UnauthorizedError,
  NoOrgError,
} from "@/server/orgs";
import { getStripe, isStripeConfigured, getPriceId } from "@/server/stripe";

const schema = z.object({
  plan: z.enum(["STARTER", "STUDIO", "AGENCY", "PAYG"]),
});

/**
 * Creates a Stripe Checkout session for the requested plan. For subscription
 * tiers the session includes both a fixed-cadence subscription item and a
 * metered overage item (price configured via `STRIPE_PRICE_*_METERED`).
 *
 * When Stripe is not configured the route returns `{ demo: true, url }` where
 * `url` is the local in-app billing page — so the UI flow remains testable
 * end-to-end without a Stripe account.
 */
export async function POST(req: Request) {
  let ctx;
  try {
    ctx = await requireOrgContext();
  } catch (e) {
    if (e instanceof UnauthorizedError) {
      return NextResponse.json({ error: "unauthenticated" }, { status: 401 });
    }
    if (e instanceof NoOrgError) {
      return NextResponse.json({ error: "no_org" }, { status: 404 });
    }
    return NextResponse.json({ error: "internal" }, { status: 500 });
  }

  const parsed = schema.safeParse(await req.json().catch(() => ({})));
  if (!parsed.success) {
    return NextResponse.json(
      { error: "invalid_request", details: parsed.error.flatten() },
      { status: 400 },
    );
  }

  const plan = parsed.data.plan;

  if (!isStripeConfigured()) {
    return NextResponse.json({
      demo: true,
      url: `/app/billing?demo_plan=${plan.toLowerCase()}`,
    });
  }

  const stripe = getStripe();
  const org = await prisma.organization.findUnique({
    where: { id: ctx.organizationId },
  });
  if (!org) {
    return NextResponse.json({ error: "no_org" }, { status: 404 });
  }

  let customerId = org.stripeCustomerId;
  if (!customerId) {
    const customer = await stripe.customers.create({
      email: org.billingEmail ?? ctx.email,
      name: org.name,
      metadata: { organizationId: org.id, ogenti_plan: plan },
    });
    customerId = customer.id;
    await prisma.organization.update({
      where: { id: org.id },
      data: { stripeCustomerId: customerId },
    });
  }

  const lineItems: Array<{ price: string; quantity?: number }> = [];

  if (plan === "PAYG") {
    const meteredPrice = getPriceId("PAYG_METERED");
    if (!meteredPrice) {
      return NextResponse.json(
        { error: "stripe_price_missing", which: "PAYG_METERED" },
        { status: 500 },
      );
    }
    lineItems.push({ price: meteredPrice });
  } else {
    const basePrice = getPriceId(plan as "STARTER" | "STUDIO" | "AGENCY");
    const overagePrice = getPriceId("OVERAGE_METERED");
    if (!basePrice) {
      return NextResponse.json(
        { error: "stripe_price_missing", which: plan },
        { status: 500 },
      );
    }
    lineItems.push({ price: basePrice, quantity: 1 });
    if (overagePrice) {
      lineItems.push({ price: overagePrice });
    }
  }

  const session = await stripe.checkout.sessions.create({
    mode: "subscription",
    customer: customerId,
    line_items: lineItems,
    success_url: absoluteUrl("/app/billing?event=success&session={CHECKOUT_SESSION_ID}"),
    cancel_url: absoluteUrl("/app/billing?event=canceled"),
    payment_method_collection: "always",
    subscription_data: {
      metadata: { organizationId: org.id, ogenti_plan: plan },
    },
    allow_promotion_codes: true,
  });

  return NextResponse.json({ url: session.url, demo: false });
}

function absoluteUrl(path: string): string {
  const base =
    process.env.NEXT_PUBLIC_APP_URL ??
    process.env.AUTH_URL ??
    "http://localhost:3000";
  return `${base}${path}`;
}
