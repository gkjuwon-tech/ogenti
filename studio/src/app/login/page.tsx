import Link from "next/link";
import { Wordmark } from "@/components/brand/Wordmark";
import { LoginForm } from "./LoginForm";
import styles from "./login.module.css";

export default function LoginPage({
  searchParams,
}: {
  searchParams?: { error?: string; callbackUrl?: string };
}) {
  return (
    <main className={styles.root}>
      <header className={styles.header}>
        <Link href="/" className={styles.brand} aria-label="Ogenti Studio">
          <Wordmark />
        </Link>
        <p className={styles.altPrompt}>
          New to Ogenti? <Link href="/signup">Request access</Link>
        </p>
      </header>

      <section className={styles.card}>
        <h1 className={styles.title}>Sign in to Ogenti Studio</h1>
        <p className={styles.subtitle}>
          Enter your work email and we&apos;ll send a sign-in link. SSO is
          available on Agency and Enterprise plans.
        </p>

        {searchParams?.error && (
          <p className={styles.errorBanner}>
            We couldn&apos;t sign you in. Please verify the email address you
            entered or contact your administrator.
          </p>
        )}

        <LoginForm callbackUrl={searchParams?.callbackUrl} />

        <p className={styles.fineprint}>
          By signing in you agree to the&nbsp;
          <Link href="/legal/terms">Terms of Service</Link> and
          acknowledge the&nbsp;
          <Link href="/legal/privacy">Privacy Notice</Link>.
        </p>
      </section>
    </main>
  );
}
