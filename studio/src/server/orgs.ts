import { prisma } from "@/lib/db";
import { auth } from "@/server/auth";

export interface ResolvedOrgContext {
  userId: string;
  email: string;
  organizationId: string;
  organizationSlug: string;
  organizationName: string;
  role: string;
}

/**
 * Resolves the active organisation for the current request. Today we pick the
 * single membership of the signed-in user; the API is shaped to accept an
 * explicit `orgSlug` later (header or path segment) once multi-org switching
 * lands in the UI.
 */
export async function requireOrgContext(): Promise<ResolvedOrgContext> {
  const session = await auth();
  if (!session?.user?.id || !session.user.email) {
    throw new UnauthorizedError("Not signed in.");
  }
  const membership = await prisma.membership.findFirst({
    where: { userId: session.user.id },
    include: { organization: true },
    orderBy: { createdAt: "asc" },
  });
  if (!membership) {
    throw new NoOrgError("No organisation found for user.");
  }
  return {
    userId: session.user.id,
    email: session.user.email,
    organizationId: membership.organizationId,
    organizationSlug: membership.organization.slug,
    organizationName: membership.organization.name,
    role: membership.role,
  };
}

/**
 * Ensures the signed-in user has a default organisation. If they don't, one
 * is created on demand using a slug derived from their email. Used by
 * onboarding and by any flow that creates resources before explicit org
 * setup (e.g. requesting access through the marketing site).
 */
export async function ensureDefaultOrg(userId: string, email: string) {
  const existing = await prisma.membership.findFirst({
    where: { userId },
    include: { organization: true },
  });
  if (existing) return existing;

  const baseSlug = slugifyEmail(email);
  const slug = await uniqueSlug(baseSlug);

  const org = await prisma.organization.create({
    data: {
      slug,
      name: defaultOrgName(email),
      billingEmail: email,
    },
  });
  return prisma.membership.create({
    data: {
      userId,
      organizationId: org.id,
      role: "OWNER",
    },
    include: { organization: true },
  });
}

function slugifyEmail(email: string): string {
  const local = email.split("@")[0] ?? "team";
  return (
    local
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 24) || "team"
  );
}

function defaultOrgName(email: string): string {
  const local = email.split("@")[0] ?? "team";
  return local.charAt(0).toUpperCase() + local.slice(1) + " Workspace";
}

async function uniqueSlug(base: string): Promise<string> {
  let candidate = base;
  let n = 1;
  // SQLite is not under contention here so a simple loop is fine.
  while (await prisma.organization.findUnique({ where: { slug: candidate } })) {
    n += 1;
    candidate = `${base}-${n}`;
  }
  return candidate;
}

export class UnauthorizedError extends Error {
  status = 401 as const;
}
export class NoOrgError extends Error {
  status = 404 as const;
}
