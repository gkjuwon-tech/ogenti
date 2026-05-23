import Stripe from "stripe";

/**
 * Stripe singleton. The SDK is initialised lazily so that local development
 * works without any Stripe keys configured. Calls that actually require
 * Stripe will throw with a clear error message if `STRIPE_SECRET_KEY` is
 * missing.
 *
 * For the demo flow, the `/api/billing/checkout` route falls back to a
 * mock checkout URL when Stripe is not configured. See its README section
 * for details.
 */

let _stripe: Stripe | null = null;

export function getStripe(): Stripe {
  if (_stripe) return _stripe;
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) {
    throw new Error(
      "STRIPE_SECRET_KEY is not configured. Set it in .env.local or fall back to demo mode.",
    );
  }
  _stripe = new Stripe(key, {
    apiVersion: "2025-02-24.acacia",
    typescript: true,
    appInfo: {
      name: "Ogenti Studio",
      version: "0.1.0",
    },
  });
  return _stripe;
}

export function isStripeConfigured(): boolean {
  return Boolean(process.env.STRIPE_SECRET_KEY);
}

/**
 * Stripe price IDs for every plan. These are populated by running
 * `pnpm tsx scripts/setup-stripe.ts` against a Stripe test account, which
 * creates Products + Prices and writes the IDs into `.env.local`.
 *
 * The MAP below describes which env var to read for each plan tier. A
 * separate metered price is used for overage / pay-as-you-go.
 */
export const STRIPE_PRICE_ENV = {
  STARTER: "STRIPE_PRICE_STARTER",
  STUDIO: "STRIPE_PRICE_STUDIO",
  AGENCY: "STRIPE_PRICE_AGENCY",
  ENTERPRISE: null,                    // negotiated; never self-serve
  PAYG_METERED: "STRIPE_PRICE_PAYG_METERED",
  OVERAGE_METERED: "STRIPE_PRICE_OVERAGE_METERED",
} as const;

export function getPriceId(
  key: keyof typeof STRIPE_PRICE_ENV,
): string | undefined {
  const envName = STRIPE_PRICE_ENV[key];
  if (!envName) return undefined;
  return process.env[envName];
}
