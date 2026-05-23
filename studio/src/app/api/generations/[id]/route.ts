import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { requireOrgContext, UnauthorizedError, NoOrgError } from "@/server/orgs";
import { getOgentiClient } from "@/server/ogenti/client";
import {
  currentRateCents,
  reportUsageEventToStripe,
} from "@/server/billing/usage";

export async function GET(
  _req: Request,
  { params }: { params: { id: string } },
) {
  let ctx;
  try {
    ctx = await requireOrgContext();
  } catch (e) {
    return mapErr(e);
  }

  const generation = await prisma.generation.findUnique({
    where: { id: params.id },
  });
  if (!generation || generation.organizationId !== ctx.organizationId) {
    return NextResponse.json({ error: "not_found" }, { status: 404 });
  }

  // Avoid hammering the inference server on terminal states.
  if (
    generation.status === "QUEUED" ||
    generation.status === "RUNNING"
  ) {
    const client = getOgentiClient();
    const remote = await client.status(generation.jobToken);
    if (remote.status === "running" && generation.status !== "RUNNING") {
      await prisma.generation.update({
        where: { id: generation.id },
        data: { status: "RUNNING", startedAt: new Date() },
      });
    } else if (remote.status === "succeeded") {
      await prisma.$transaction(async (tx) => {
        const updated = await tx.generation.update({
          where: { id: generation.id },
          data: {
            status: "SUCCEEDED",
            startedAt: generation.startedAt ?? new Date(),
            finishedAt: new Date(),
            resultUrl: remote.resultUrl,
            thumbnailUrl: remote.thumbnailUrl,
            computeSeconds: remote.computeSeconds,
            billedSeconds: generation.durationSeconds,
          },
        });
        const rate = await currentRateCents(generation.organizationId);
        const event = await tx.usageEvent.create({
          data: {
            organizationId: generation.organizationId,
            generationId: updated.id,
            idempotencyKey: `gen:${updated.id}`,
            billedSeconds: generation.durationSeconds,
            ratePerSecondCents: rate,
            costCents: rate * generation.durationSeconds,
            description: `Ogenti generation ${updated.id}`,
          },
        });
        // Fire-and-forget the metered usage record; failure here is recovered
        // by the periodic reconciliation worker (TODO).
        reportUsageEventToStripe(event.id).catch((err) => {
          console.error("usage report failed", err);
        });
      });
    } else if (remote.status === "failed") {
      await prisma.generation.update({
        where: { id: generation.id },
        data: {
          status: "FAILED",
          finishedAt: new Date(),
          errorMessage: remote.errorMessage ?? "Unknown error",
        },
      });
    } else if (remote.status === "canceled") {
      await prisma.generation.update({
        where: { id: generation.id },
        data: { status: "CANCELED", finishedAt: new Date() },
      });
    }
  }

  const fresh = await prisma.generation.findUnique({
    where: { id: params.id },
  });
  return NextResponse.json({ generation: fresh });
}

export async function DELETE(
  _req: Request,
  { params }: { params: { id: string } },
) {
  let ctx;
  try {
    ctx = await requireOrgContext();
  } catch (e) {
    return mapErr(e);
  }

  const generation = await prisma.generation.findUnique({
    where: { id: params.id },
  });
  if (!generation || generation.organizationId !== ctx.organizationId) {
    return NextResponse.json({ error: "not_found" }, { status: 404 });
  }

  if (generation.status === "QUEUED" || generation.status === "RUNNING") {
    const client = getOgentiClient();
    await client.cancel(generation.jobToken).catch(() => {});
    await prisma.generation.update({
      where: { id: generation.id },
      data: { status: "CANCELED", finishedAt: new Date() },
    });
  }
  return NextResponse.json({ ok: true });
}

function mapErr(e: unknown) {
  if (e instanceof UnauthorizedError) {
    return NextResponse.json({ error: "unauthenticated" }, { status: 401 });
  }
  if (e instanceof NoOrgError) {
    return NextResponse.json({ error: "no_org" }, { status: 404 });
  }
  console.error(e);
  return NextResponse.json({ error: "internal" }, { status: 500 });
}
