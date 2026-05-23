import styles from "./Why.module.css";

interface Pillar {
  id: string;
  number: string;
  heading: string;
  blurb: string;
  detail: string;
}

const pillars: Pillar[] = [
  {
    id: "glyph",
    number: "01",
    heading: "Type-safe glyphs",
    blurb:
      "Brand wordmarks survive the diffusion process intact. Logos render in the typeface you typed — not the generic AI substitute.",
    detail:
      "A dedicated glyph branch is fused into the DiT backbone with a separate gating ladder, so spelling, kerning, and stroke-weight remain stable across frames.",
  },
  {
    id: "anatomy",
    number: "02",
    heading: "Anatomy-locked humans",
    blurb:
      "Hand counts, joint orientations, and bilateral symmetry stay legal. No six-finger model, no inverted elbows, no melted faces.",
    detail:
      "An anatomical consistency prior is attached as an auxiliary loss with zero-init gates, so the model only learns to correct anatomy — never to destroy it.",
  },
  {
    id: "physics",
    number: "03",
    heading: "Physically grounded motion",
    blurb:
      "Liquids pour, fabrics drape, glass reflects, smoke rises. Pre-simulated PyBullet keyframes condition the generation directly.",
    detail:
      "Per-clip physics descriptors are baked into the conditioner. The model treats Newtonian dynamics as evidence, not as a stylistic hint to discard mid-sequence.",
  },
];

export function Why() {
  return (
    <section className={styles.root} aria-labelledby="why-heading">
      <div className={`container ${styles.inner}`}>
        <header className={styles.header}>
          <p className="eyebrow">Why Ogenti</p>
          <h2 id="why-heading" className={styles.title}>
            Three failure modes
            <br />
            we engineered out.
          </h2>
          <p className={styles.lede}>
            Every other AI video model fails the same three checks when an
            agency producer screens a take. Ogenti rebuilds the backbone where
            those failures live.
          </p>
        </header>

        <ol className={styles.grid}>
          {pillars.map((p) => (
            <li key={p.id} className={styles.card}>
              <header className={styles.cardHead}>
                <span className={`mono ${styles.cardNumber}`}>{p.number}</span>
                <h3 className={styles.cardHeading}>{p.heading}</h3>
              </header>
              <p className={styles.cardBlurb}>{p.blurb}</p>
              <p className={styles.cardDetail}>{p.detail}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
