import { redirect } from "next/navigation";
import { auth } from "@/server/auth";
import { ensureDefaultOrg, requireOrgContext } from "@/server/orgs";
import { prisma } from "@/lib/db";
import { Sidebar } from "@/components/console/Sidebar";
import styles from "./layout.module.css";

export const dynamic = "force-dynamic";

export default async function AppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await auth();
  if (!session?.user?.id || !session.user.email) {
    redirect("/login?callbackUrl=/app");
  }

  await ensureDefaultOrg(session.user.id, session.user.email);

  const ctx = await requireOrgContext();
  const org = await prisma.organization.findUnique({
    where: { id: ctx.organizationId },
  });

  return (
    <div className={styles.root}>
      <Sidebar
        orgName={ctx.organizationName}
        orgSlug={ctx.organizationSlug}
        planTier={org?.planTier ?? "PAYG"}
        userEmail={ctx.email}
      />
      <main className={styles.main}>{children}</main>
    </div>
  );
}
