/**
 * Stripe product + price bootstrap script.
 *
 * Creates Products for each plan tier and corresponding Prices. Prints the
 * resulting price IDs in the format expected by `.env.local`. Run against a
 * test-mode Stripe account first:
 *
 *   STRIPE_SECRET_KEY=sk_test_xxx pnpm tsx scripts/setup-stripe.ts
 *
 * Re-running the script is idempotent: it searches Stripe by metadata key
 * `ogenti_plan` before creating new products.
 */

import Stripe from "stripe";
import { PAYG_RATE_USD, PLANS } from "../src/lib/pricing";

const key = process.env.STRIPE_SECRET_KEY;
if (!key) {
  console.error("STRIPE_SECRET_KEY is required.");
  process.exit(1);
}

const stripe = new Stripe(key, {
  apiVersion: "2025-02-24.acacia",
  typescript: true,
});

const usdCents = (usd: number) => Math.round(usd * 100);

async function findOrCreateProduct(metadataKey: string, name: string) {
  const search = await stripe.products.search({
    query: `metadata['ogenti_plan']:'${metadataKey}'`,
    limit: 1,
  });
  if (search.data[0]) return search.data[0];
  return stripe.products.create({
    name,
    metadata: { ogenti_plan: metadataKey },
  });
}

async function findOrCreateRecurringPrice(
  product: Stripe.Product,
  amountUsdCents: number,
  nickname: string,
) {
  const prices = await stripe.prices.list({
    product: product.id,
    active: true,
    limit: 10,
  });
  const existing = prices.data.find(
    (p) =>
      p.unit_amount === amountUsdCents &&
      p.recurring?.interval === "month" &&
      p.recurring?.usage_type === "licensed",
  );
  if (existing) return existing;
  return stripe.prices.create({
    product: product.id,
    unit_amount: amountUsdCents,
    currency: "usd",
    recurring: { interval: "month", usage_type: "licensed" },
    nickname,
  });
}

async function findOrCreateMeteredPrice(
  product: Stripe.Product,
  unitAmountCents: number,
  nickname: string,
) {
  const prices = await stripe.prices.list({
    product: product.id,
    active: true,
    limit: 10,
  });
  const existing = prices.data.find(
    (p) =>
      p.unit_amount === unitAmountCents &&
      p.recurring?.usage_type === "metered",
  );
  if (existing) return existing;
  return stripe.prices.create({
    product: product.id,
    unit_amount: unitAmountCents,
    currency: "usd",
    recurring: {
      interval: "month",
      usage_type: "metered",
      aggregate_usage: "sum",
    },
    nickname,
  });
}

async function main() {
  const env: Record<string, string> = {};

  for (const plan of PLANS) {
    if (plan.id === "enterprise") continue;
    const product = await findOrCreateProduct(
      plan.id.toUpperCase(),
      `Ogenti Studio · ${plan.name}`,
    );
    if (plan.monthlyUsd != null && plan.monthlyUsd > 0) {
      const price = await findOrCreateRecurringPrice(
        product,
        usdCents(plan.monthlyUsd),
        `${plan.name} monthly`,
      );
      env[`STRIPE_PRICE_${plan.id.toUpperCase()}`] = price.id;
    }
  }

  const overageProduct = await findOrCreateProduct(
    "OVERAGE",
    "Ogenti Studio · Overage seconds",
  );
  const overagePrice = await findOrCreateMeteredPrice(
    overageProduct,
    usdCents(PAYG_RATE_USD),
    "Overage metered ($/s)",
  );
  env.STRIPE_PRICE_OVERAGE_METERED = overagePrice.id;

  const paygProduct = await findOrCreateProduct(
    "PAYG",
    "Ogenti Studio · Pay-as-you-go",
  );
  const paygPrice = await findOrCreateMeteredPrice(
    paygProduct,
    usdCents(PAYG_RATE_USD),
    "Pay-as-you-go metered ($/s)",
  );
  env.STRIPE_PRICE_PAYG_METERED = paygPrice.id;

  console.log("\n# Add these to your .env.local");
  console.log("# ------------------------------");
  for (const [k, v] of Object.entries(env)) {
    console.log(`${k}=${v}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
