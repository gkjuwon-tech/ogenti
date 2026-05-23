import Link from "next/link";
import { Wordmark } from "@/components/brand/Wordmark";
import { LoginForm } from "@/app/login/LoginForm";
import styles from "@/app/login/login.module.css";

export default function SignupPage() {
  return (
    <main className={styles.root}>
      <header className={styles.header}>
        <Link href="/" className={styles.brand} aria-label="Ogenti Studio">
          <Wordmark />
        </Link>
        <p className={styles.altPrompt}>
          Already have an account? <Link href="/login">Sign in</Link>
        </p>
      </header>

      <section className={styles.card}>
        <h1 className={styles.title}>Request access to Ogenti Studio</h1>
        <p className={styles.subtitle}>
          Ogenti is in private beta. Submit your work email and a member of
          our team will follow up within one business day with a tenant
          provisioned to your domain.
        </p>

        <LoginForm />

        <p className={styles.fineprint}>
          Need to chat first? Reach us at&nbsp;
          <a href="mailto:hello@ogenti.dev">hello@ogenti.dev</a>.
        </p>
      </section>
    </main>
  );
}
