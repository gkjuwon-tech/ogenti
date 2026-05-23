import { ButtonLink } from "@/components/ui/Button";
import styles from "./Hero.module.css";

export function Hero() {
  return (
    <section className={styles.root}>
      <div className={`container ${styles.inner}`}>
        <div className={styles.copy}>
          <p className={`mono ${styles.dateline}`}>
            <span>OG&middot;001</span>
            <span aria-hidden>·</span>
            <span>Private beta &mdash; Q2 cohort</span>
          </p>

          <h1 className={styles.title}>
            A video model built for the people who get fired when the wordmark
            is wrong.
          </h1>

          <p className={styles.lede}>
            Ogenti is a structural retrofit of a 14&#8209;billion-parameter
            video diffusion transformer. We replaced the parts that fail in
            front of a producer &mdash; glyphs, anatomy, and Newtonian motion
            &mdash; without disturbing the parts that already work. The
            result is footage that survives the room where the cut is
            reviewed.
          </p>

          <div className={styles.ctaRow}>
            <ButtonLink href="/signup" size="lg" variant="primary">
              Request access
            </ButtonLink>
            <ButtonLink href="/docs/brief" size="lg" variant="link">
              Read the technical brief
            </ButtonLink>
          </div>
        </div>

        <aside className={styles.spec} aria-label="Model specification">
          <div className={styles.specHead}>
            <span className={`mono ${styles.specHeadKey}`}>Model</span>
            <span className={`mono ${styles.specHeadValue}`}>
              Ogenti&nbsp;v0.6 &middot; Wan&nbsp;2.2 retrofit (A14B&nbsp;MoE)
            </span>
          </div>

          <dl className={styles.specList}>
            <div className={styles.specRow}>
              <dt>Backbone</dt>
              <dd>14 B parameter video DiT (MoE high&#8209;noise / low&#8209;noise experts)</dd>
            </div>
            <div className={styles.specRow}>
              <dt>Resolution</dt>
              <dd>up to 4 K, 24 fps</dd>
            </div>
            <div className={styles.specRow}>
              <dt>Clip length</dt>
              <dd>2 &ndash; 20 sec, deterministic seed</dd>
            </div>
            <div className={styles.specRow}>
              <dt>Aspect ratios</dt>
              <dd>16 : 9 &middot; 9 : 16 &middot; 1 : 1 &middot; 4 : 5</dd>
            </div>
            <div className={styles.specRow}>
              <dt>Retrofit ladder</dt>
              <dd>five phases &middot; 4 800 step total &middot; zero-init gates</dd>
            </div>
            <div className={styles.specRow}>
              <dt>Conditioners</dt>
              <dd>brand glyphs &middot; anatomy keypoints &middot; PyBullet physics</dd>
            </div>
            <div className={styles.specRow}>
              <dt>Step 0 invariant</dt>
              <dd>bit&#8209;equivalent to vanilla Wan 2.2</dd>
            </div>
            <div className={styles.specRow}>
              <dt>Method</dt>
              <dd>
                published as <span className="mono">RFC&nbsp;0001 &mdash; 0006</span>
              </dd>
            </div>
          </dl>
        </aside>
      </div>

      <div className={`container ${styles.logoRow}`}>
        <dl className={styles.pilot}>
          <div className={styles.pilotRow}>
            <dt className={`mono ${styles.pilotKey}`}>Pilot</dt>
            <dd className={styles.pilotValue}>
              Closed cohort. Brand and agency teams under NDA.
            </dd>
          </div>
          <div className={styles.pilotRow}>
            <dt className={`mono ${styles.pilotKey}`}>Cities</dt>
            <dd className={`mono ${styles.pilotValue}`}>
              Seoul &middot; Tokyo &middot; New&nbsp;York &middot; London
            </dd>
          </div>
          <div className={styles.pilotRow}>
            <dt className={`mono ${styles.pilotKey}`}>Window</dt>
            <dd className={`mono ${styles.pilotValue}`}>
              Q2 &mdash; rolling intake, weekly
            </dd>
          </div>
        </dl>
      </div>
    </section>
  );
}
