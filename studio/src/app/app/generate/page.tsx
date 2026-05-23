import { GenerateClient } from "./GenerateClient";
import { prisma } from "@/lib/db";
import { requireOrgContext } from "@/server/orgs";
import { currentRateCents } from "@/server/billing/usage";

export const dynamic = "force-dynamic";

export default async function GeneratePage() {
  const ctx = await requireOrgContext();
  const recent = await prisma.generation.findMany({
    where: { organizationId: ctx.organizationId },
    orderBy: { createdAt: "desc" },
    take: 8,
  });
  const rate = await currentRateCents(ctx.organizationId);
  return (
    <GenerateClient
      ratePerSecondCents={rate}
      initialRecent={recent.map((g) => ({
        id: g.id,
        prompt: g.prompt,
        status: g.status,
        durationSeconds: g.durationSeconds,
        resolution: g.resolution,
        createdAt: g.createdAt.toISOString(),
        resultUrl: g.resultUrl,
        thumbnailUrl: g.thumbnailUrl,
      }))}
    />
  );
}
