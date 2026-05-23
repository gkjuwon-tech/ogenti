import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { requireOrgContext, UnauthorizedError, NoOrgError } from "@/server/orgs";
import { getOgentiClient, type GenerateRequest } from "@/server/ogenti/client";
import { currentRateCents } from "@/server/billing/usage";

const createSchema = z.object({
  prompt: z.string().min(1).max(2000),
  negativePrompt: z.string().max(2000).optional(),
  aspectRatio: z.enum(["16:9", "9:16", "1:1", "4:5"]).default("16:9"),
  durationSeconds: z.number().int().min(1).max(20).default(4),
  resolution: z.enum(["720p", "1080p", "4k"]).default("1080p"),
  seed: z.number().int().optional(),
  referenceUrl: z.string().url().optional(),
});

export async function POST(req: Request) {
  let ctx;
  try {
    ctx = await requireOrgContext();
  } catch (e) {
    return errorResponse(e);
  }

  const body = await req.json().catch(() => null);
  const parsed = createSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "invalid_request", details: parsed.error.flatten() },
      { status: 400 },
    );
  }

  const org = await prisma.organization.findUnique({
    where: { id: ctx.organizationId },
  });
  if (!org) {
    return NextResponse.json({ error: "organization_missing" }, { status: 404 });
  }
  if (org.generationPaused) {
    return NextResponse.json(
      {
        error: "billing_blocked",
        message:
          "Generation is paused for this organisation due to an unpaid invoice. Update payment to resume.",
      },
      { status: 402 },
    );
  }

  const client = getOgentiClient();
  const request: GenerateRequest = {
    prompt: parsed.data.prompt,
    negativePrompt: parsed.data.negativePrompt,
    aspectRatio: parsed.data.aspectRatio,
    durationSeconds: parsed.data.durationSeconds,
    resolution: parsed.data.resolution,
    seed: parsed.data.seed,
    referenceUrl: parsed.data.referenceUrl,
  };

  const { jobToken } = await client.submit(request);

  const generation = await prisma.generation.create({
    data: {
      organizationId: ctx.organizationId,
      createdByUserId: ctx.userId,
      prompt: request.prompt,
      negativePrompt: request.negativePrompt,
      aspectRatio: request.aspectRatio,
      durationSeconds: request.durationSeconds,
      resolution: request.resolution,
      seed: request.seed,
      referenceUrl: request.referenceUrl,
      jobToken,
      status: "QUEUED",
    },
  });

  return NextResponse.json(
    {
      id: generation.id,
      status: generation.status,
      jobToken,
      durationSeconds: generation.durationSeconds,
      estimatedCostCents:
        generation.durationSeconds * (await currentRateCents(org.id)),
    },
    { status: 201 },
  );
}

export async function GET() {
  let ctx;
  try {
    ctx = await requireOrgContext();
  } catch (e) {
    return errorResponse(e);
  }

  const rows = await prisma.generation.findMany({
    where: { organizationId: ctx.organizationId },
    orderBy: { createdAt: "desc" },
    take: 50,
  });
  return NextResponse.json({ generations: rows });
}

function errorResponse(e: unknown) {
  if (e instanceof UnauthorizedError) {
    return NextResponse.json({ error: "unauthenticated" }, { status: 401 });
  }
  if (e instanceof NoOrgError) {
    return NextResponse.json({ error: "no_org" }, { status: 404 });
  }
  console.error(e);
  return NextResponse.json({ error: "internal" }, { status: 500 });
}
