import Link from "next/link";
import { Wordmark } from "@/components/brand/Wordmark";
import styles from "./Footer.module.css";

const groups: Array<{
  heading: string;
  items: Array<{ label: string; href: string }>;
}> = [
  {
    heading: "Product",
    items: [
      { label: "Overview", href: "/product" },
      { label: "Studio console", href: "/app" },
      { label: "API", href: "/docs/api" },
      { label: "Changelog", href: "/changelog" },
      { label: "Status", href: "/status" },
    ],
  },
  {
    heading: "Solutions",
    items: [
      { label: "For brand teams", href: "/solutions/brand" },
      { label: "For agencies", href: "/solutions/agencies" },
      { label: "For studios", href: "/solutions/studios" },
      { label: "For platforms", href: "/solutions/platforms" },
    ],
  },
  {
    heading: "Resources",
    items: [
      { label: "Research", href: "/research" },
      { label: "Technical brief", href: "/docs/brief" },
      { label: "Customer stories", href: "/customers" },
      { label: "Press kit", href: "/press" },
    ],
  },
  {
    heading: "Company",
    items: [
      { label: "About", href: "/about" },
      { label: "Careers", href: "/careers" },
      { label: "Security", href: "/security" },
      { label: "Contact sales", href: "/contact" },
    ],
  },
  {
    heading: "Legal",
    items: [
      { label: "Terms of service", href: "/legal/terms" },
      { label: "Privacy", href: "/legal/privacy" },
      { label: "Acceptable use", href: "/legal/aup" },
      { label: "DPA", href: "/legal/dpa" },
    ],
  },
];

export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className={styles.root}>
      <div className={`container ${styles.inner}`}>
        <div className={styles.brandCol}>
          <Wordmark />
          <p className={styles.tagline}>
            Advertising-grade AI video for teams that ship for a living.
          </p>
          <p className={styles.meta}>
            <span>Built in Seoul.</span>
            <span aria-hidden> · </span>
            <span>Compute in US-East and EU-West.</span>
          </p>
        </div>

        <nav className={styles.grid} aria-label="Sitemap">
          {groups.map((g) => (
            <div key={g.heading} className={styles.group}>
              <h3 className={styles.groupHeading}>{g.heading}</h3>
              <ul className={styles.list}>
                {g.items.map((item) => (
                  <li key={item.href}>
                    <Link href={item.href} className={styles.link}>
                      {item.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
      </div>

      <div className={`container ${styles.legalRow}`}>
        <p className={styles.copyright}>
          © {year} Ogenti, Inc. All rights reserved.
        </p>
        <p className={styles.fineprint}>
          <span>Ogenti, Ogenti Studio, and the concentric mark are trademarks of Ogenti, Inc.</span>
        </p>
      </div>
    </footer>
  );
}
