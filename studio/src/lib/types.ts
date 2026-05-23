/**
 * String-union types for fields that would have been Prisma enums on a
 * provider that supports them (Postgres, MySQL). Use these everywhere a
 * Prisma row's status field is consumed.
 */

export type PlanTier =
  | "STARTER"
  | "STUDIO"
  | "AGENCY"
  | "ENTERPRISE"
  | "PAYG";

export type SubscriptionStatus =
  | "TRIALING"
  | "ACTIVE"
  | "PAST_DUE"
  | "CANCELED"
  | "INCOMPLETE"
  | "PAUSED";

export type MembershipRole = "OWNER" | "ADMIN" | "MEMBER" | "VIEWER";

export type GenerationStatusDb =
  | "QUEUED"
  | "RUNNING"
  | "SUCCEEDED"
  | "FAILED"
  | "CANCELED";

export type InvoiceStatus =
  | "DRAFT"
  | "OPEN"
  | "PAID"
  | "VOID"
  | "UNCOLLECTIBLE";
