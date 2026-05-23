"use client";

import { useState } from "react";
import { Button } from "@/components/ui/Button";

interface Props {
  planTier: string;
  hasStripeCustomer: boolean;
  switchTo?: "STARTER" | "STUDIO" | "AGENCY" | "PAYG";
  isCurrent?: boolean;
}

export function BillingActions({
  planTier,
  hasStripeCustomer,
  switchTo,
  isCurrent,
}: Props) {
  const [busy, setBusy] = useState(false);

  async function startCheckout(plan: string) {
    setBusy(true);
    const resp = await fetch("/api/billing/checkout", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ plan }),
    });
    setBusy(false);
    if (!resp.ok) {
      const text = await resp.text();
      alert(`Could not start checkout: ${text}`);
      return;
    }
    const data = (await resp.json()) as { url: string };
    window.location.href = data.url;
  }

  async function openPortal() {
    setBusy(true);
    const resp = await fetch("/api/billing/portal", { method: "POST" });
    setBusy(false);
    if (!resp.ok) {
      const text = await resp.text();
      alert(`Could not open portal: ${text}`);
      return;
    }
    const data = (await resp.json()) as { url: string };
    window.location.href = data.url;
  }

  if (switchTo) {
    if (isCurrent) {
      return (
        <span
          style={{
            fontSize: 12,
            color: "var(--ink-500)",
            padding: "8px 12px",
          }}
        >
          Current plan
        </span>
      );
    }
    return (
      <Button
        variant="secondary"
        size="sm"
        onClick={() => startCheckout(switchTo)}
        disabled={busy}
      >
        {busy ? "Loading…" : "Switch"}
      </Button>
    );
  }

  return (
    <div style={{ display: "flex", gap: 8 }}>
      {hasStripeCustomer ? (
        <Button
          variant="secondary"
          size="sm"
          onClick={openPortal}
          disabled={busy}
        >
          {busy ? "Loading…" : "Manage in Stripe"}
        </Button>
      ) : (
        <Button
          variant="primary"
          size="sm"
          onClick={() => startCheckout(planTier === "PAYG" ? "PAYG" : "STUDIO")}
          disabled={busy}
        >
          {busy ? "Loading…" : "Add payment method"}
        </Button>
      )}
    </div>
  );
}
