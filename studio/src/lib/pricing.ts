/**
 * Pricing source of truth. Used by the marketing pricing table, the in-app
 * plan switcher, and the Stripe seed script.
 *
 * Numbers here are PROPOSALS to validate against real inference cost once the
 * Ogenti inference server is online. The pay-as-you-go rate represents the
 * single floor below which we will not price even bespoke deals.
 */

export type PlanId = "starter" | "studio" | "agency" | "enterprise";

export interface Plan {
  id: PlanId;
  name: string;
  tagline: string;
  monthlyUsd: number | null; // null = contact sales
  includedSeconds: number; // monthly included generation seconds
  overageRateUsd: number; // $ per generation second beyond included
  seats: number | "unlimited";
  features: string[];
  highlight?: boolean;
  cta: { label: string; href: string };
}

export const PAYG_RATE_USD = 1.5;

export const PLANS: Plan[] = [
  {
    id: "starter",
    name: "Starter",
    tagline:
      "For independent creatives validating Ogenti against existing pipelines.",
    monthlyUsd: 99,
    includedSeconds: 60,
    overageRateUsd: 1.2,
    seats: 2,
    features: [
      "Up to 1080p, 4-second clips",
      "Glyph branch + anatomy losses",
      "PNG storyboards + MP4 export",
      "Email support, 2-business-day SLA",
    ],
    cta: { label: "Start with Starter", href: "/signup?plan=starter" },
  },
  {
    id: "studio",
    name: "Studio",
    tagline:
      "For in-house brand studios producing campaigns end-to-end on Ogenti.",
    monthlyUsd: 499,
    includedSeconds: 360,
    overageRateUsd: 0.95,
    seats: 8,
    features: [
      "Up to 4K, 12-second clips",
      "Reference upload + style controls",
      "Project workspaces & shared library",
      "Slack-based support, 1-business-day SLA",
      "Audit log export",
    ],
    highlight: true,
    cta: { label: "Start with Studio", href: "/signup?plan=studio" },
  },
  {
    id: "agency",
    name: "Agency",
    tagline:
      "For creative agencies running multiple brand accounts in parallel.",
    monthlyUsd: 1999,
    includedSeconds: 1800,
    overageRateUsd: 0.75,
    seats: 25,
    features: [
      "Up to 4K, 20-second clips",
      "Multi-tenant client workspaces",
      "Brand glyph allow-list & lock",
      "Dedicated CSM, 4-hour priority SLA",
      "SSO (Okta, Entra, Google)",
      "Quarterly model fine-tuning slot",
    ],
    cta: { label: "Start with Agency", href: "/signup?plan=agency" },
  },
  {
    id: "enterprise",
    name: "Enterprise",
    tagline:
      "For global brand holdcos and platforms with bespoke deployment needs.",
    monthlyUsd: null,
    includedSeconds: 0,
    overageRateUsd: 0,
    seats: "unlimited",
    features: [
      "Custom volume commitments",
      "Dedicated tenancy or VPC peering",
      "DPA + SCC, EU/US/APAC residency",
      "24×7 incident SLA",
      "Named solutions engineer",
      "Co-research on private RFCs",
    ],
    cta: { label: "Contact sales", href: "/contact" },
  },
];

export function formatUsd(amount: number, opts?: { fractional?: boolean }) {
  const fractional = opts?.fractional ?? amount % 1 !== 0;
  return amount.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: fractional ? 2 : 0,
    maximumFractionDigits: fractional ? 2 : 0,
  });
}
