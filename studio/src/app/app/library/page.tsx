import Link from "next/link";
import { prisma } from "@/lib/db";
import { requireOrgContext } from "@/server/orgs";
import { formatRelative } from "@/lib/format";
import styles from "./library.module.css";

export const dynamic = "force-dynamic";

export default async function LibraryPage() {
  const ctx = await requireOrgContext();
  const rows = await prisma.generation.findMany({
    where: { organizationId: ctx.organizationId },
    orderBy: { createdAt: "desc" },
    take: 100,
  });

  return (
    <>
      <header className={styles.header}>
        <div>
          <p className="eyebrow">Library</p>
          <h1 className={styles.title}>Past generations</h1>
          <p className={styles.subtitle}>
            All renders for this workspace. Stored for 90 days on the Starter
            plan, indefinitely on Studio and above.
          </p>
        </div>
        <Link href="/app/generate" className={styles.newLink}>
          + New generation
        </Link>
      </header>

      {rows.length === 0 ? (
        <p className={styles.empty}>
          Your library will appear here once you queue your first render.
          <br />
          <Link href="/app/generate" className={styles.emptyLink}>
            Go to Generate →
          </Link>
        </p>
      ) : (
        <ul className={styles.grid}>
          {rows.map((g) => (
            <li key={g.id} className={styles.card}>
              <div className={styles.thumb} aria-hidden>
                <span className={styles.thumbInner} data-status={g.status}>
                  {g.status === "SUCCEEDED" ? (
                    <span className={styles.playIcon}>▶</span>
                  ) : (
                    <span className={styles.statusOnly}>
                      {prettyStatus(g.status)}
                    </span>
                  )}
                </span>
              </div>
              <div className={styles.cardBody}>
                <p className={styles.prompt} title={g.prompt}>
                  {g.prompt.length > 80
                    ? g.prompt.slice(0, 80) + "…"
                    : g.prompt}
                </p>
                <p className={styles.meta}>
                  <span className="mono">{g.resolution}</span>
                  <span aria-hidden> · </span>
                  <span className="mono">{g.durationSeconds}s</span>
                  <span aria-hidden> · </span>
                  <span>{formatRelative(g.createdAt)}</span>
                </p>
                <div className={styles.cardFooter}>
                  <span
                    className={styles.statusPill}
                    data-status={g.status}
                  >
                    {prettyStatus(g.status)}
                  </span>
                  {g.resultUrl && (
                    <Link
                      href={g.resultUrl}
                      className={styles.download}
                      prefetch={false}
                    >
                      Download
                    </Link>
                  )}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

function prettyStatus(s: string): string {
  switch (s) {
    case "QUEUED":
      return "Queued";
    case "RUNNING":
      return "Rendering";
    case "SUCCEEDED":
      return "Ready";
    case "FAILED":
      return "Failed";
    case "CANCELED":
      return "Canceled";
    default:
      return s;
  }
}
