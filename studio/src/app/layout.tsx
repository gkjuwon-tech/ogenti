import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "@/styles/globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
  variable: "--font-inter",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
  variable: "--font-jetbrains",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://studio.ogenti.com"),
  title: {
    default: "Ogenti Studio — Advertising-grade AI video",
    template: "%s · Ogenti Studio",
  },
  description:
    "Type-safe glyphs, anatomy-locked humans, physically-grounded motion. The AI video model for agencies and brand teams who can't ship AI tells.",
  applicationName: "Ogenti Studio",
  keywords: [
    "AI video",
    "advertising AI",
    "text to video",
    "Wan2.2 retrofit",
    "Ogenti",
    "B2B video generation",
  ],
  openGraph: {
    type: "website",
    title: "Ogenti Studio",
    description:
      "Advertising-grade AI video. Without the AI tells.",
    siteName: "Ogenti Studio",
  },
  twitter: {
    card: "summary_large_image",
  },
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrains.variable}`}>
      <body>{children}</body>
    </html>
  );
}
