import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import {
  requireOrgContext,
  UnauthorizedError,
  NoOrgError,
} from "@/server/orgs";
import { getStripe, isStripeConfigured } from "@/server/stripe";

export async function POST() {
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

  if (!isStripeConfigured()) {
    return NextResponse.json({ demo: true, url: "/app/billing?demo_portal=1" });
  }

  const stripe = getStripe();
  const org = await prisma.organization.findUnique({
    where: { id: ctx.organizationId },
  });
  if (!org?.stripeCustomerId) {
    return NextResponse.json(
      { error: "no_stripe_customer" },
      { status: 400 },
    );
  }
  const portal = await stripe.billingPortal.sessions.create({
    customer: org.stripeCustomerId,
    return_url: absoluteUrl("/app/billing"),
  });
  return NextResponse.json({ url: portal.url, demo: false });
}

function absoluteUrl(path: string): string {
  const base =
    process.env.NEXT_PUBLIC_APP_URL ??
    process.env.AUTH_URL ??
    "http://localhost:3000";
  return `${base}${path}`;
}
