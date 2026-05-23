"use client";

import { useState } from "react";
import { signIn } from "next-auth/react";
import { Button } from "@/components/ui/Button";
import styles from "./login.module.css";

interface Props {
  callbackUrl?: string;
}

export function LoginForm({ callbackUrl }: Props) {
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [info, setInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setInfo(null);
    setError(null);

    const looksValid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    if (!looksValid) {
      setError("Please enter a valid work email.");
      setSubmitting(false);
      return;
    }

    // Try the dev-login (always available in non-production builds first).
    const devResp = await signIn("dev-login", {
      email,
      redirect: false,
      callbackUrl: callbackUrl ?? "/app",
    });
    if (devResp?.ok) {
      window.location.href = devResp.url ?? "/app";
      return;
    }

    // Fall back to magic link via Resend (production / preview).
    const resp = await signIn("resend", {
      email,
      redirect: false,
      callbackUrl: callbackUrl ?? "/app",
    });
    if (resp?.ok) {
      setInfo("Check your inbox — we just sent a sign-in link.");
    } else {
      setError(
        "Magic-link sign-in is not yet configured on this environment. Try the dev login by entering your email above.",
      );
    }
    setSubmitting(false);
  }

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <label className={styles.field}>
        <span>Work email</span>
        <input
          type="email"
          name="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@studio.example"
          className={styles.input}
        />
      </label>

      <Button
        type="submit"
        size="lg"
        variant="primary"
        disabled={submitting}
      >
        {submitting ? "Sending…" : "Continue with email"}
      </Button>

      {info && <p className={styles.info}>{info}</p>}
      {error && <p className={styles.error}>{error}</p>}
    </form>
  );
}
