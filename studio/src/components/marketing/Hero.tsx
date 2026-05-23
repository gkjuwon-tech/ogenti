import { ButtonLink } from "@/components/ui/Button";
import styles from "./Hero.module.css";

export function Hero() {
  return (
    <section className={styles.root}>
      <div className={`container ${styles.inner}`}>
        <div className={styles.copy}>
          <p className={`eyebrow ${styles.eyebrow}`}>
            <span className={styles.dot} aria-hidden />
            Now in private beta · Apply for early access
          </p>

          <h1 className={styles.title}>
            Advertising-grade AI video.
            <br />
            <span className={styles.titleAccent}>Without the AI tells.</span>
          </h1>

          <p className={styles.lede}>
            Ogenti is a structurally retrofit video foundation model built for
            brands and agencies. Type-safe glyphs, anatomy-locked humans,
            physically grounded motion — every output passes the&nbsp;
            <em>is&#8209;this&#8209;AI?</em> sniff test that kills today&apos;s
            generative pipelines.
          </p>

          <div className={styles.ctaRow}>
            <ButtonLink href="/signup" size="lg" variant="primary">
              Request access
            </ButtonLink>
            <ButtonLink href="/docs/brief" size="lg" variant="secondary">
              Read the technical brief
              <span aria-hidden> →</span>
            </ButtonLink>
          </div>

          <ul className={styles.bullets}>
            <li>
              <span className={styles.bulletLabel}>OCR-stable</span>
              <span className={styles.bulletValue}>brand glyph branch</span>
            </li>
            <li>
              <span className={styles.bulletLabel}>Anatomy-aware</span>
              <span className={styles.bulletValue}>21k human-keyframe prior</span>
            </li>
            <li>
              <span className={styles.bulletLabel}>Physics-conditioned</span>
              <span className={styles.bulletValue}>PyBullet pre-simulation</span>
            </li>
          </ul>
        </div>

        <aside className={styles.viewer} aria-label="Example output (placeholder)">
          <div className={styles.viewerInner}>
            <div className={styles.viewerTopBar}>
              <span className={styles.viewerDot} data-tone="red" />
              <span className={styles.viewerDot} data-tone="yellow" />
              <span className={styles.viewerDot} data-tone="green" />
              <span className={`mono ${styles.viewerTitle}`}>
                studio.ogenti.com / preview
              </span>
            </div>
            <div className={styles.viewerStage}>
              <div className={styles.gradientFrame} aria-hidden />
              <div className={styles.frameOverlay}>
                <span className={`eyebrow ${styles.frameLabel}`}>brief</span>
                <p className={styles.framePrompt}>
                  &ldquo;Slow push-in on a perfume bottle on warm marble, glass
                  refracts soft window light, label reads&nbsp;
                  <span className={`mono ${styles.glyph}`}>AURELIA Nº7</span>&nbsp;
                  in clean serif, 4&nbsp;sec, 24&nbsp;fps.&rdquo;
                </p>
              </div>
              <dl className={styles.frameMeta}>
                <div>
                  <dt>glyph integrity</dt>
                  <dd className="mono">0.987</dd>
                </div>
                <div>
                  <dt>anatomy score</dt>
                  <dd className="mono">n/a</dd>
                </div>
                <div>
                  <dt>physics fit</dt>
                  <dd className="mono">0.962</dd>
                </div>
                <div>
                  <dt>render time</dt>
                  <dd className="mono">38.4s</dd>
                </div>
              </dl>
            </div>
          </div>
        </aside>
      </div>

      <div className={`container ${styles.logoRow}`}>
        <p className={styles.logoLabel}>
          Built on research published as RFC&nbsp;0001 — RFC&nbsp;0006. Trusted
          by independents and Fortune&nbsp;500 brand teams in closed pilot.
        </p>
        <ul className={styles.logoStrip}>
          {[
            "monolith",
            "concentric",
            "axiom",
            "pylon",
            "harbour & co",
            "northscale",
          ].map((name) => (
            <li key={name} className={styles.logo}>
              {name}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
