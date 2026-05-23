"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Wordmark } from "@/components/brand/Wordmark";
import styles from "./Sidebar.module.css";

interface NavItem {
  href: string;
  label: string;
  icon: React.ReactNode;
}

const items: NavItem[] = [
  { href: "/app", label: "Overview", icon: <DashIcon /> },
  { href: "/app/generate", label: "Generate", icon: <GenIcon /> },
  { href: "/app/library", label: "Library", icon: <LibIcon /> },
  { href: "/app/billing", label: "Billing", icon: <BillIcon /> },
  { href: "/app/settings", label: "Settings", icon: <SettingsIcon /> },
];

interface SidebarProps {
  orgName: string;
  orgSlug: string;
  planTier: string;
  userEmail: string;
}

export function Sidebar({ orgName, orgSlug, planTier, userEmail }: SidebarProps) {
  const pathname = usePathname();
  return (
    <aside className={styles.root} aria-label="Workspace navigation">
      <div className={styles.brandRow}>
        <Wordmark variant="compact" />
        <span className={`mono ${styles.envTag}`}>console</span>
      </div>

      <div className={styles.orgRow}>
        <p className={styles.orgName}>{orgName}</p>
        <p className={styles.orgMeta}>
          <span className={`mono ${styles.orgSlug}`}>{orgSlug}</span>
          <span aria-hidden> · </span>
          <span className={styles.plan}>{prettyPlan(planTier)}</span>
        </p>
      </div>

      <nav className={styles.nav}>
        {items.map((item) => {
          const active =
            pathname === item.href ||
            (item.href !== "/app" && pathname?.startsWith(item.href + "/"));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={styles.navLink}
              data-active={active ? "true" : undefined}
            >
              <span className={styles.icon} aria-hidden>
                {item.icon}
              </span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className={styles.footer}>
        <div className={styles.userRow}>
          <span className={styles.avatar} aria-hidden>
            {userEmail.charAt(0).toUpperCase()}
          </span>
          <div className={styles.userText}>
            <span className={styles.userEmail}>{userEmail}</span>
            <Link href="/api/auth/signout" className={styles.signOut}>
              Sign out
            </Link>
          </div>
        </div>
      </div>
    </aside>
  );
}

function prettyPlan(tier: string): string {
  switch (tier) {
    case "STARTER":
      return "Starter";
    case "STUDIO":
      return "Studio";
    case "AGENCY":
      return "Agency";
    case "ENTERPRISE":
      return "Enterprise";
    case "PAYG":
    default:
      return "Pay-as-you-go";
  }
}

function DashIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none">
      <rect x="2" y="2" width="5" height="5" stroke="currentColor" strokeWidth="1.2" />
      <rect x="9" y="2" width="5" height="5" stroke="currentColor" strokeWidth="1.2" />
      <rect x="2" y="9" width="5" height="5" stroke="currentColor" strokeWidth="1.2" />
      <rect x="9" y="9" width="5" height="5" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}
function GenIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none">
      <path d="M3 4h10v8H3z" stroke="currentColor" strokeWidth="1.2" />
      <path d="M7 6v4l3-2z" fill="currentColor" />
    </svg>
  );
}
function LibIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none">
      <path d="M3 3h3v10H3zM7 3h3v10H7zM11 3h2v10h-2z" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}
function BillIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none">
      <path d="M2 4h12v8H2z" stroke="currentColor" strokeWidth="1.2" />
      <path d="M2 7h12" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}
function SettingsIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none">
      <circle cx="8" cy="8" r="2" stroke="currentColor" strokeWidth="1.2" />
      <path
        d="M8 1v2M8 13v2M3.5 3.5l1.5 1.5M11 11l1.5 1.5M1 8h2M13 8h2M3.5 12.5L5 11M11 5l1.5-1.5"
        stroke="currentColor"
        strokeWidth="1.2"
      />
    </svg>
  );
}
