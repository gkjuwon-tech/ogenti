import { prisma } from "@/lib/db";
import { requireOrgContext } from "@/server/orgs";
import { formatRelative } from "@/lib/format";
import styles from "./settings.module.css";

export const dynamic = "force-dynamic";

export default async function SettingsPage() {
  const ctx = await requireOrgContext();
  const [org, memberships, apiKeys] = await Promise.all([
    prisma.organization.findUnique({ where: { id: ctx.organizationId } }),
    prisma.membership.findMany({
      where: { organizationId: ctx.organizationId },
      include: { user: true },
      orderBy: { createdAt: "asc" },
    }),
    prisma.apiKey.findMany({
      where: { organizationId: ctx.organizationId },
      orderBy: { createdAt: "desc" },
      take: 10,
    }),
  ]);
  if (!org) return null;

  return (
    <>
      <header className={styles.header}>
        <p className="eyebrow">Settings</p>
        <h1 className={styles.title}>Workspace</h1>
        <p className={styles.subtitle}>
          Organisation profile, team members, API keys, and webhook endpoints.
        </p>
      </header>

      <section className={styles.card}>
        <header className={styles.cardHead}>
          <h2 className={styles.sectionTitle}>Organisation</h2>
        </header>
        <dl className={styles.descList}>
          <div>
            <dt>Name</dt>
            <dd>{org.name}</dd>
          </div>
          <div>
            <dt>Slug</dt>
            <dd className="mono">{org.slug}</dd>
          </div>
          <div>
            <dt>Legal name</dt>
            <dd>{org.legalName ?? <span className={styles.muted}>—</span>}</dd>
          </div>
          <div>
            <dt>Billing email</dt>
            <dd className="mono">{org.billingEmail ?? "—"}</dd>
          </div>
          <div>
            <dt>Plan</dt>
            <dd>{org.planTier}</dd>
          </div>
          <div>
            <dt>Subscription status</dt>
            <dd>{org.subscriptionStatus}</dd>
          </div>
        </dl>
      </section>

      <section className={styles.card}>
        <header className={styles.cardHead}>
          <h2 className={styles.sectionTitle}>Team members</h2>
          <p className={styles.sectionSub}>
            SSO and SCIM are available on the Agency plan and above.
          </p>
        </header>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Email</th>
              <th>Role</th>
              <th>Joined</th>
            </tr>
          </thead>
          <tbody>
            {memberships.map((m) => (
              <tr key={m.id}>
                <td className="mono">{m.user.email}</td>
                <td>{prettyRole(m.role)}</td>
                <td className="mono">{formatRelative(m.createdAt)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className={styles.card}>
        <header className={styles.cardHead}>
          <h2 className={styles.sectionTitle}>API keys</h2>
          <p className={styles.sectionSub}>
            Use API keys to submit generation jobs from your own systems.
            Available on Studio and above.
          </p>
        </header>
        {apiKeys.length === 0 ? (
          <p className={styles.empty}>
            No API keys yet. Once an admin creates one, the prefix is shown
            here for reference; the full key is only shown once at creation.
          </p>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Label</th>
                <th>Prefix</th>
                <th>Created</th>
                <th>Last used</th>
              </tr>
            </thead>
            <tbody>
              {apiKeys.map((k) => (
                <tr key={k.id}>
                  <td>{k.label}</td>
                  <td className="mono">{k.prefix}…</td>
                  <td className="mono">{formatRelative(k.createdAt)}</td>
                  <td className="mono">
                    {k.lastUsedAt ? formatRelative(k.lastUsedAt) : "never"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className={styles.card}>
        <header className={styles.cardHead}>
          <h2 className={styles.sectionTitle}>Webhooks</h2>
          <p className={styles.sectionSub}>
            We&apos;ll POST signed events to a URL of your choice when a
            generation finishes. Configure endpoints from your custom
            integration when ready.
          </p>
        </header>
        <p className={styles.empty}>
          Webhook configuration UI ships with the API GA milestone.
        </p>
      </section>
    </>
  );
}

function prettyRole(role: string): string {
  return role.charAt(0) + role.slice(1).toLowerCase();
}
