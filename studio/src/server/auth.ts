import NextAuth, { type DefaultSession } from "next-auth";
import Credentials from "next-auth/providers/credentials";
import Resend from "next-auth/providers/resend";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { prisma } from "@/lib/db";

/**
 * Ogenti Studio auth.
 *
 * Two providers:
 *   - Resend magic link (production / preview)
 *   - Credentials with a passwordless dev login (local development only) —
 *     gated behind NODE_ENV !== "production", so a hostile production deploy
 *     cannot accidentally enable it.
 *
 * The Prisma adapter is wired directly to our SQLite (or future Postgres)
 * database. Session strategy is JWT for fast middleware checks; durable
 * session rows live in `Session` for the magic-link provider's flow only.
 */

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
    } & DefaultSession["user"];
  }
}

const isProd = process.env.NODE_ENV === "production";

export const { handlers, auth, signIn, signOut } = NextAuth({
  adapter: PrismaAdapter(prisma),
  session: { strategy: "jwt" },
  pages: {
    signIn: "/login",
    verifyRequest: "/login/verify",
    newUser: "/onboarding",
  },
  providers: [
    ...(process.env.AUTH_RESEND_KEY
      ? [
          Resend({
            apiKey: process.env.AUTH_RESEND_KEY,
            from:
              process.env.AUTH_EMAIL_FROM ?? "Ogenti Studio <login@ogenti.dev>",
          }),
        ]
      : []),
    ...(!isProd
      ? [
          Credentials({
            id: "dev-login",
            name: "Developer Login",
            credentials: {
              email: { label: "Email", type: "email" },
            },
            async authorize(credentials) {
              const email = String(credentials?.email ?? "").trim().toLowerCase();
              if (!email || !email.includes("@")) return null;
              const user = await prisma.user.upsert({
                where: { email },
                update: {},
                create: { email, name: email.split("@")[0] },
              });
              return {
                id: user.id,
                email: user.email,
                name: user.name ?? undefined,
              };
            },
          }),
        ]
      : []),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.sub = user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (token.sub) {
        session.user.id = token.sub;
      }
      return session;
    },
  },
});
