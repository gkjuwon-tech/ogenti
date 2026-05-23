import { ButtonLink } from "@/components/ui/Button";
import styles from "./CtaBand.module.css";

export function CtaBand() {
  return (
    <section className={styles.root}>
      <div className={`container ${styles.inner}`}>
        <div className={styles.copy}>
          <p className="eyebrow">Get started</p>
          <h2 className={styles.title}>
            Bring Ogenti into your next pitch this week.
          </h2>
          <p className={styles.lede}>
            Request access and a member of our team will provision a tenant
            within one business day. Onboarding includes a brand-glyph upload,
            a sample brief workshop, and a billing walkthrough.
          </p>
        </div>
        <div className={styles.actions}>
          <ButtonLink href="/signup" variant="primary" size="lg">
            Request access
          </ButtonLink>
          <ButtonLink href="/contact" variant="secondary" size="lg">
            Talk to sales
          </ButtonLink>
        </div>
      </div>
    </section>
  );
}
