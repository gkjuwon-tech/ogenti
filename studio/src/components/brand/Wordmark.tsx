import styles from "./Wordmark.module.css";

interface WordmarkProps {
  variant?: "default" | "compact";
  tone?: "ink" | "inverse";
}

export function Wordmark({ variant = "default", tone = "ink" }: WordmarkProps) {
  return (
    <span
      className={styles.root}
      data-variant={variant}
      data-tone={tone}
      aria-label="Ogenti Studio"
    >
      <span className={styles.mark} aria-hidden>
        <svg viewBox="0 0 24 24" width="22" height="22">
          {/* Concentric stencil glyph — represents the "O" of Ogenti and the
              identity-anchored attention loop. Sharp 1px stroke, no gradient. */}
          <rect
            x="1"
            y="1"
            width="22"
            height="22"
            rx="3"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.4"
          />
          <path
            d="M 7 7 L 17 7 L 17 17 L 7 17 Z M 10 10 L 14 10 L 14 14 L 10 14 Z"
            fill="currentColor"
            fillRule="evenodd"
          />
        </svg>
      </span>
      {variant === "default" ? (
        <span className={styles.word}>
          Ogenti<span className={styles.suffix}>&nbsp;Studio</span>
        </span>
      ) : (
        <span className={styles.word}>Ogenti</span>
      )}
    </span>
  );
}
