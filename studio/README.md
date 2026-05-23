# Ogenti Studio

The B2B SaaS console for [Ogenti](../README.md) — a structurally retrofit
video foundation model built for brands and agencies. Studio is the public
landing page, the in-app generation console, and the billing infrastructure
in a single Next.js 14 application.

Studio is intentionally model-agnostic. The inference server is plugged in
behind an `OgentiClient` interface in `src/server/ogenti/client.ts`; until
that server is online, a `MockOgentiClient` produces deterministic fake
progress for end-to-end UI testing.

## Stack

- Next.js 14 (App Router) + React 18 + TypeScript strict
- Vanilla CSS Modules (no Tailwind, no UI library, no `classnames` package)
  with a typed design-token system in `src/styles/tokens.css`
- Prisma ORM with SQLite locally; switch the provider to Postgres for
  production without changing application code
- NextAuth v5 — Resend magic-link in production, dev-login credentials in
  non-production so the console is reachable on `localhost:3000`
- Stripe Billing — recurring subscriptions plus metered usage records for
  pay-as-you-go and overage, with a signature-verifying webhook handler
- Zod for request validation on every API route

## Architecture

```
studio/
├─ prisma/schema.prisma           # multi-tenant schema (User/Org/Membership/...)
├─ src/
│  ├─ app/
│  │  ├─ page.tsx                 # landing page composition
│  │  ├─ login/  signup/          # auth flow
│  │  ├─ app/                     # console (auth-protected)
│  │  │  ├─ layout.tsx            # sidebar + org context
│  │  │  ├─ page.tsx              # dashboard
│  │  │  ├─ generate/             # prompt UI + mock client
│  │  │  ├─ library/              # past generations
│  │  │  ├─ billing/              # cycle, plans, invoices
│  │  │  └─ settings/             # org, team, API keys, webhooks
│  │  └─ api/
│  │     ├─ auth/[...nextauth]/   # NextAuth handlers
│  │     ├─ generations/          # job submit + status
│  │     └─ billing/              # checkout, portal, webhook
│  ├─ components/
│  │  ├─ brand/                   # Wordmark
│  │  ├─ ui/                      # Button (and friends)
│  │  ├─ marketing/               # landing sections
│  │  └─ console/                 # sidebar
│  ├─ server/
│  │  ├─ auth.ts                  # NextAuth config
│  │  ├─ orgs.ts                  # org context resolver
│  │  ├─ stripe.ts                # Stripe singleton + price ID map
│  │  ├─ billing/                 # cycle + metered usage
│  │  └─ ogenti/client.ts         # Mock & Http inference clients
│  ├─ lib/                        # db, pricing source-of-truth, format
│  └─ styles/                     # tokens.css + globals.css
└─ scripts/setup-stripe.ts        # idempotent product/price seed
```

## Quick start

```bash
pnpm install
cp .env.example .env.local
# fill in AUTH_SECRET at minimum (Stripe + Resend can stay blank for demo)
pnpm db:push
pnpm dev
```

Then open <http://localhost:3000>. To enter the console:

1. Click **Sign in** in the top right (or visit `/login`).
2. Enter any email — the dev-login credentials provider will create the
   user and a default organisation on first sign-in.

Billing remains in **demo mode** until you configure Stripe — checkout
buttons redirect back to the billing page with a `demo_plan=` query
parameter so the flow can be exercised end-to-end without a Stripe key.

## Configuring Stripe (test mode)

```bash
# 1) Add your test-mode key to .env.local
echo 'STRIPE_SECRET_KEY=sk_test_…' >> .env.local

# 2) Bootstrap products and prices
pnpm tsx scripts/setup-stripe.ts >> .env.local

# 3) Forward webhooks to the local app
stripe listen --forward-to localhost:3000/api/billing/webhook
# copy the printed signing secret into STRIPE_WEBHOOK_SECRET in .env.local
```

After this, the checkout buttons on `/app/billing` open real Stripe Checkout
sessions and the webhook handler will mirror subscription and invoice state
into the local database.

## Plugging in the real Ogenti model

```ts
// .env.local
OGENTI_INFERENCE_URL=https://inference.ogenti.dev
OGENTI_INFERENCE_TOKEN=ogt_live_…
```

`getOgentiClient()` in `src/server/ogenti/client.ts` detects these and swaps
the `MockOgentiClient` for the `HttpOgentiClient`. No UI code changes.

The expected inference contract is documented inline at the top of
`client.ts` — `POST /v1/generations`, `GET /v1/generations/{token}`,
`POST /v1/generations/{token}/cancel`.

## Pricing

Source of truth lives in `src/lib/pricing.ts` (consumed by the marketing
pricing table, the in-app plan switcher, and the Stripe seed script). The
proposal:

| Tier        | Monthly | Included        | Overage     |
|-------------|---------|-----------------|-------------|
| Starter     | $99     | 60 s            | $1.20/s     |
| Studio      | $499    | 360 s           | $0.95/s     |
| Agency      | $1,999  | 1,800 s         | $0.75/s     |
| Enterprise  | Custom  | Negotiated      | Negotiated  |
| Pay-as-you-go | $0    | None            | $1.50/s     |

These numbers assume an inference cost floor of roughly $3–4 per generated
minute on rented A100 capacity and bake in a 70–80% gross margin. Re-tune
once the production inference server emits real cost telemetry.

## Design system

No Tailwind, no UI library. Everything is composed from:

- **Tokens** — `src/styles/tokens.css` defines colour, type, spacing,
  radius, motion and depth on a strict 4 px grid. Single accent
  (`--accent-600: #1e4dff`), hairline borders, mono numerals.
- **CSS Modules** — every component owns a `Foo.module.css` next to its
  `Foo.tsx`. Variants and sizes are exposed via `data-variant` /
  `data-size` attributes so consumers stay TypeScript-typed.
- **Type ramp** — Inter for sans, JetBrains Mono for numerals (loaded by
  `next/font`).

If you change a token, change it in `tokens.css`. Do not introduce ad-hoc
colour or spacing values inside components.

## Scripts

| Command            | What it does                                |
|--------------------|---------------------------------------------|
| `pnpm dev`         | Next.js dev server with HMR                 |
| `pnpm build`       | Production build                            |
| `pnpm start`       | Run the production build                    |
| `pnpm lint`        | ESLint (Next.js core-web-vitals)            |
| `pnpm db:generate` | Prisma client codegen                       |
| `pnpm db:push`     | Sync the SQLite schema with Prisma          |

## What this is NOT

- It does not contain the model weights, training code, or inference
  server. Those live in the parent `ogenti/` package.
- It does not include image or video generation today. The console
  produces deterministic placeholder MP4 URLs from the mock client.
- It does not handle invoicing in arrears outside of Stripe. Period
  rollover, dunning, retries, and PDF generation are all delegated to
  Stripe Billing.
