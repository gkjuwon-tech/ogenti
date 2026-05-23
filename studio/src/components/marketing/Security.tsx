import styles from "./Security.module.css";

const items = [
  {
    title: "Tenant isolation",
    body: "Each organisation lives in its own logical workspace with row-level access enforcement. No shared model fine-tunes, ever.",
  },
  {
    title: "Audit log streaming",
    body: "Every generation, every login, every billing event is captured and exportable to your SIEM via signed JSONL.",
  },
  {
    title: "Data residency",
    body: "Choose US-East or EU-West for compute and storage. Enterprise contracts can add an APAC region under a private agreement.",
  },
  {
    title: "Compliance posture",
    body: "SOC 2 Type I in progress with audit start date of Q1. DPA + SCC available on request. Public security page coming alongside.",
  },
  {
    title: "Encryption at rest and in transit",
    body: "TLS 1.3 in transit, AES-256 at rest, customer-managed keys available on the Enterprise plan via AWS KMS or Azure Key Vault.",
  },
  {
    title: "Responsible use",
    body: "An always-on content filter prevents likeness misuse, regulated-product violations, and unauthorised brand impersonation.",
  },
];

export function Security() {
  return (
    <section className={styles.root} aria-labelledby="security-heading">
      <div className={`container ${styles.inner}`}>
        <header className={styles.header}>
          <p className="eyebrow">Trust</p>
          <h2 id="security-heading" className={styles.title}>
            Production controls before you ship a frame.
          </h2>
          <p className={styles.lede}>
            Ogenti was built for paying clients from week one. Security
            posture, residency choices, and audit hooks are not roadmap
            promises — they ship with the product.
          </p>
        </header>

        <ul className={styles.grid}>
          {items.map((item) => (
            <li key={item.title} className={styles.cell}>
              <h3 className={styles.cellTitle}>{item.title}</h3>
              <p className={styles.cellBody}>{item.body}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
