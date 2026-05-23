import styles from "./Audiences.module.css";

const audiences = [
  {
    id: "brand",
    label: "Brand teams",
    headline: "Concept films that don't betray the wordmark.",
    bullets: [
      "Locked-glyph rendering for every logo in your brand book",
      "Reference-image upload with style retention across shots",
      "Approval workflow with audit trail per take",
    ],
  },
  {
    id: "agency",
    label: "Agencies",
    headline: "Internal pre-vis at the speed of a pitch deck.",
    bullets: [
      "Multi-tenant client workspaces with strict isolation",
      "Per-account brand asset locking and usage attribution",
      "White-label export for client preview portals",
    ],
  },
  {
    id: "studio",
    label: "Studios",
    headline: "Producer-grade output without the AI giveaway.",
    bullets: [
      "Direct EXR-style metadata on every render for the grade",
      "Anatomy and physics scorecards bundled with every clip",
      "Sidecar masks for compositing in Resolve and Nuke",
    ],
  },
  {
    id: "platform",
    label: "Platforms",
    headline: "Ship Ogenti generation to your own users.",
    bullets: [
      "REST + webhook API with idempotent job submission",
      "Bring-your-own-bucket storage for tenant assets",
      "Pricing wholesale, billed on a single monthly invoice",
    ],
  },
];

export function Audiences() {
  return (
    <section className={styles.root} aria-labelledby="audiences-heading">
      <div className={`container ${styles.inner}`}>
        <header className={styles.header}>
          <p className="eyebrow">Built for</p>
          <h2 id="audiences-heading" className={styles.title}>
            One model, four buyers.
          </h2>
        </header>

        <ul className={styles.list}>
          {audiences.map((a) => (
            <li key={a.id} className={styles.row}>
              <div className={styles.rowLabel}>
                <span className={`mono ${styles.rowKey}`}>{a.id}</span>
                <span className={styles.rowAudience}>{a.label}</span>
              </div>
              <div className={styles.rowCopy}>
                <h3 className={styles.rowHeadline}>{a.headline}</h3>
                <ul className={styles.bullets}>
                  {a.bullets.map((b) => (
                    <li key={b}>{b}</li>
                  ))}
                </ul>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
