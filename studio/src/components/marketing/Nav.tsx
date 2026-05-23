import Link from "next/link";
import { Wordmark } from "@/components/brand/Wordmark";
import { ButtonLink } from "@/components/ui/Button";
import styles from "./Nav.module.css";

export function Nav() {
  return (
    <header className={styles.root}>
      <div className={`container ${styles.inner}`}>
        <Link href="/" className={styles.brandLink} aria-label="Ogenti Studio home">
          <Wordmark />
        </Link>

        <nav className={styles.links} aria-label="Primary">
          <Link href="/product" className={styles.link}>
            Product
          </Link>
          <Link href="/pricing" className={styles.link}>
            Pricing
          </Link>
          <Link href="/customers" className={styles.link}>
            Customers
          </Link>
          <Link href="/docs" className={styles.link}>
            Docs
          </Link>
          <Link href="/research" className={styles.link}>
            Research
          </Link>
        </nav>

        <div className={styles.actions}>
          <Link href="/login" className={styles.link} data-discreet>
            Log in
          </Link>
          <ButtonLink href="/signup" variant="primary" size="sm">
            Request access
          </ButtonLink>
        </div>
      </div>
    </header>
  );
}
