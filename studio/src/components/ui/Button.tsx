import * as React from "react";
import styles from "./Button.module.css";

type ButtonVariant = "primary" | "secondary" | "ghost" | "link";
type ButtonSize = "sm" | "md" | "lg";

interface ButtonOwnProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  trailing?: React.ReactNode;
  leading?: React.ReactNode;
}

type ButtonProps = ButtonOwnProps &
  React.ButtonHTMLAttributes<HTMLButtonElement>;
type AnchorProps = ButtonOwnProps &
  React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string };

function classNames(...xs: Array<string | undefined | false>): string {
  return xs.filter(Boolean).join(" ");
}

export function Button({
  variant = "primary",
  size = "md",
  trailing,
  leading,
  className,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={classNames(styles.root, className)}
      data-variant={variant}
      data-size={size}
      {...rest}
    >
      {leading && <span className={styles.affix}>{leading}</span>}
      <span className={styles.label}>{children}</span>
      {trailing && <span className={styles.affix}>{trailing}</span>}
    </button>
  );
}

export function ButtonLink({
  variant = "primary",
  size = "md",
  trailing,
  leading,
  className,
  children,
  ...rest
}: AnchorProps) {
  return (
    <a
      className={classNames(styles.root, className)}
      data-variant={variant}
      data-size={size}
      {...rest}
    >
      {leading && <span className={styles.affix}>{leading}</span>}
      <span className={styles.label}>{children}</span>
      {trailing && <span className={styles.affix}>{trailing}</span>}
    </a>
  );
}
