import styles from "./Pipeline.module.css";

const stages = [
  {
    code: "Phase 1",
    title: "Entity & glyph init",
    detail: "entity bank · glyph branch · camera motion embed · 1000 steps",
  },
  {
    code: "Phase 2",
    title: "Subject motion & anatomy",
    detail: "subject motion embed · anatomy loss · 800 steps",
  },
  {
    code: "Phase 3",
    title: "OCR hardening",
    detail: "glyph gate fine-tune · 700 steps",
  },
  {
    code: "Phase 4",
    title: "Realism — the AI-tell killer",
    detail:
      "skin · material · motion blur · film grain · lens artifacts · 1500 steps",
  },
  {
    code: "Phase 5",
    title: "Physics keyframes",
    detail:
      "physics keyframe embed · PyBullet pre-simulation · 800 steps",
  },
];

export function Pipeline() {
  return (
    <section className={styles.root} aria-labelledby="pipeline-heading">
      <div className={`container ${styles.inner}`}>
        <header className={styles.header}>
          <p className="eyebrow">The retrofit ladder</p>
          <h2 id="pipeline-heading" className={styles.title}>
            Five locked phases.
            <br />
            <span className={styles.titleAccent}>
              One invariant — step zero is bit-equivalent to vanilla.
            </span>
          </h2>
          <p className={styles.lede}>
            Ogenti is a <em>structural retrofit</em> of an open video DiT.
            Every add-on is zero-init gated, so at training step zero the model
            produces the same outputs as the base. Each phase unfreezes a
            specific module, adds a specific loss, and only progresses when
            the previous gate has converged.
          </p>
        </header>

        <ol className={styles.ladder}>
          {stages.map((s, i) => (
            <li key={s.code} className={styles.row}>
              <div className={styles.tickColumn}>
                <span className={`mono ${styles.tickLabel}`}>{s.code}</span>
                <span className={styles.tick} aria-hidden />
              </div>
              <div className={styles.detail}>
                <h3 className={styles.rowTitle}>{s.title}</h3>
                <p className={`mono ${styles.rowMeta}`}>{s.detail}</p>
              </div>
              <span className={`mono ${styles.index}`}>
                {String(i + 1).padStart(2, "0")} / 05
              </span>
            </li>
          ))}
        </ol>

        <div className={styles.note}>
          <p>
            Detailed methodology lives in our public technical brief
            (RFC&nbsp;0001 — RFC&nbsp;0006) and is reviewed by an external
            committee of three retired CG supervisors and one cinematographer.
          </p>
        </div>
      </div>
    </section>
  );
}
