import { Nav } from "@/components/marketing/Nav";
import { Hero } from "@/components/marketing/Hero";
import { Why } from "@/components/marketing/Why";
import { Pipeline } from "@/components/marketing/Pipeline";
import { Audiences } from "@/components/marketing/Audiences";
import { Pricing } from "@/components/marketing/Pricing";
import { Security } from "@/components/marketing/Security";
import { CtaBand } from "@/components/marketing/CtaBand";
import { Footer } from "@/components/marketing/Footer";

export default function HomePage() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Why />
        <Pipeline />
        <Audiences />
        <Pricing />
        <Security />
        <CtaBand />
      </main>
      <Footer />
    </>
  );
}
