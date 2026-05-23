import Link from "next/link";
import { ButtonLink } from "@/components/ui/Button";
import { PAYG_RATE_USD, PLANS, formatUsd } from "@/lib/pricing";
import styles from "./Pricing.module.css";

export function Pricing() {
  return (
    <section className={styles.root} aria-labelledby="pricing-heading">
      <div className={`container ${styles.inner}`}>
        <header className={styles.header}>
          <p className="eyebrow">Pricing</p>
          <h2 id="pricing-heading" className={styles.title}>
            Pricing built for production.
          </h2>
          <p className={styles.lede}>
            Monthly subscriptions for predictable budgeting plus pay-as-you-go
            billed in arrears for spillover. No prepaid credits, no per-seat
            traps. Invoiced in USD on net-30 for Agency and above.
          </p>
        </header>

        <div className={styles.grid}>
          {PLANS.map((plan) => (
            <article
              key={plan.id}
              className={styles.card}
              data-highlight={plan.highlight ? "true" : undefined}
            >
              <header className={styles.cardHead}>
                <h3 className={styles.planName}>{plan.name}</h3>
                {plan.highlight && (
                  <span className={styles.badge}>Most popular</span>
                )}
              </header>
              <p className={styles.planTagline}>{plan.tagline}</p>

              <div className={styles.priceRow}>
                {plan.monthlyUsd === null ? (
                  <span className={`mono ${styles.priceCustom}`}>Custom</span>
                ) : (
                  <>
                    <span className={`mono ${styles.priceAmount}`}>
                      {formatUsd(plan.monthlyUsd)}
                    </span>
                    <span className={styles.priceCadence}>
                      <span>/ month</span>
                      <span>billed monthly</span>
                    </span>
                  </>
                )}
              </div>

              <dl className={styles.specs}>
                <div>
                  <dt>Included</dt>
                  <dd className="mono">
                    {plan.includedSeconds > 0
                      ? `${plan.includedSeconds.toLocaleString()} s / mo`
                      : "Negotiated"}
                  </dd>
                </div>
                <div>
                  <dt>Overage</dt>
                  <dd className="mono">
                    {plan.monthlyUsd === null
                      ? "Negotiated"
                      : `${formatUsd(plan.overageRateUsd, { fractional: true })} / s`}
                  </dd>
                </div>
                <div>
                  <dt>Seats</dt>
                  <dd className="mono">
                    {plan.seats === "unlimited"
                      ? "Unlimited"
                      : `${plan.seats}`}
                  </dd>
                </div>
              </dl>

              <ul className={styles.features}>
                {plan.features.map((f) => (
                  <li key={f}>
                    <svg
                      viewBox="0 0 16 16"
                      width="14"
                      height="14"
                      aria-hidden
                      className={styles.tick}
                    >
                      <path
                        d="M3.5 8.5 L7 12 L13 5"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinecap="square"
                      />
                    </svg>
                    <span>{f}</span>
                  </li>
                ))}
              </ul>

              <ButtonLink
                href={plan.cta.href}
                variant={plan.highlight ? "primary" : "secondary"}
                size="md"
                className={styles.cta}
              >
                {plan.cta.label}
              </ButtonLink>
            </article>
          ))}
        </div>

        <aside className={styles.payg}>
          <div>
            <p className="eyebrow">Pay-as-you-go</p>
            <h3 className={styles.paygTitle}>
              No commitment, billed monthly in arrears.
            </h3>
            <p className={styles.paygCopy}>
              Skip the subscription. We charge your card or invoice your AP
              department at the end of each calendar month for every second of
              video you generated. Idempotency keys, audit trail, and itemised
              invoice exports included.
            </p>
          </div>
          <div className={styles.paygPrice}>
            <span className={`mono ${styles.paygAmount}`}>
              {formatUsd(PAYG_RATE_USD, { fractional: true })}
            </span>
            <span className={styles.paygUnit}>per generation second</span>
            <Link href="/signup?plan=payg" className={styles.paygLink}>
              Activate pay-as-you-go <span aria-hidden>→</span>
            </Link>
          </div>
        </aside>
      </div>
    </section>
  );
}
