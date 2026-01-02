import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { cn } from "@/lib/utils";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";

export default function About() {
  const [activeSection, setActiveSection] = useState("");

  const navItems = useMemo(
    () => [
      { id: "overview", label: "Overview" },
      { id: "timeline", label: "Timeline", indent: true },
      { id: "contributing", label: "Contributing" },
      { id: "credits", label: "Credits" },
    ],
    [],
  );

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (!el) return;

    const prefersReducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

    el.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth" });
    setActiveSection(id);
    window.history.replaceState(null, "", `#${encodeURIComponent(id)}`);
  };

  useEffect(() => {
    const syncFromHash = () => {
      let id = "";
      try {
        id = decodeURIComponent(window.location.hash.replace(/^#/, ""));
      } catch {
        return;
      }
      if (!id) return;
      scrollToSection(id);
    };

    // Initial
    const timer = window.setTimeout(syncFromHash, 0);

    // Subsequent (e.g. user edits hash / copy-pastes anchors)
    window.addEventListener("hashchange", syncFromHash);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("hashchange", syncFromHash);
    };
  }, []);

  const ComingSoon = ({ title }: { title: string }) => (
    <Card className="border-border/70 bg-card/70 p-5">
      <div className="text-sm font-medium text-foreground">{title}</div>
      <p className="mt-1 text-sm text-muted-foreground prose max-w-none">
        This section is being written. If you’d like to help,{" "}
        <a
          href="https://github.com/PlatosTwin/pandects-app/issues"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Open an issue (opens in a new tab)"
        >
          open an issue
        </a>{" "}
        with suggestions.
      </p>
    </Card>
  );

  return (
    <PageShell
      size="xl"
      title="About"
      subtitle="Project background, data sources, and how the dataset is assembled."
    >
      <div className="grid gap-8 lg:grid-cols-[280px_1fr]">
        <aside className="hidden lg:block">
          <div className="sticky top-20">
            <Card className="border-border/70 bg-background/70 p-3 backdrop-blur">
              <div className="px-2 pb-2 pt-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                On this page
              </div>
              <nav aria-label="About page sections">
                <ul className="space-y-1">
                  {navItems.map(({ id, label, indent }) => (
                    <li key={id} className={indent ? "ml-4" : undefined}>
                      <button
                        type="button"
                        onClick={() => scrollToSection(id)}
                        aria-current={activeSection === id ? "location" : undefined}
                        aria-controls={id}
                        className={cn(
                          "w-full rounded-md px-3 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                          activeSection === id
                            ? "bg-primary/10 text-primary font-medium"
                            : "text-muted-foreground hover:bg-accent hover:text-foreground",
                        )}
                      >
                        {label}
                      </button>
                    </li>
                  ))}
                </ul>
              </nav>
            </Card>
          </div>
        </aside>

        <div className="space-y-12">
          <section id="overview" className="scroll-mt-24 space-y-4">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Overview
            </h2>
            <p className="max-w-3xl text-muted-foreground">
              Pandects is an open-source M&A research platform built to make it
              easier to browse and analyze clauses across definitive agreements.
            </p>

            <div id="timeline" className="scroll-mt-24 pt-2">
              <h3 className="text-lg font-semibold text-foreground">
                Timeline
              </h3>
              <dl className="mt-4 space-y-4 pl-6">
                <div className="grid grid-cols-[160px_1fr] gap-x-6">
                  <dt className="font-semibold text-foreground">
                    May – Jul ’25
                  </dt>
                  <dd className="text-muted-foreground">
                    Preliminary testing; proof of concept
                  </dd>
                </div>
                <div className="grid grid-cols-[160px_1fr] gap-x-6">
                  <dt className="font-semibold text-foreground">Jul ’25</dt>
                  <dd className="text-muted-foreground">
                    Basic UI development
                  </dd>
                </div>
                <div className="grid grid-cols-[160px_1fr] gap-x-6">
                  <dt className="font-semibold text-foreground">
                    Aug – Nov ’25
                  </dt>
                  <dd className="text-muted-foreground prose max-w-none">
                    <Link to="/feedback">
                      End‑user surveys
                    </Link>{" "}
                    and pipeline development
                  </dd>
                </div>
                <div className="grid grid-cols-[160px_1fr] gap-x-6">
                  <dt className="font-semibold text-foreground">
                    Nov ’25 – Jan ’26
                  </dt>
                  <dd className="text-muted-foreground">
                    Pipeline refinement; testing
                  </dd>
                </div>
                <div className="grid grid-cols-[160px_1fr] gap-x-6">
                  <dt className="font-semibold text-foreground">Feb ’26</dt>
                  <dd className="text-muted-foreground">Soft launch</dd>
                </div>
              </dl>
            </div>
          </section>

          <section id="contributing" className="scroll-mt-24 space-y-4">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Contributing
            </h2>
            <p className="text-muted-foreground prose max-w-none">
              This is an open-source project, and contributions are welcome.
              See the{" "}
              <a
                href="https://github.com/PlatosTwin/pandects-app"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="GitHub repository (opens in a new tab)"
              >
                GitHub repository
              </a>{" "}
              for details.
            </p>
            <p className="text-muted-foreground">
              We are especially thankful for focused contributions to:
            </p>
            <ul className="list-disc space-y-2 pl-6 text-muted-foreground">
              <li>Security hardening, including OAuth flow reviews.</li>
              <li>Accessibility improvements (keyboard, focus, contrast).</li>
              <li>ML model optimization.</li>
            </ul>
          </section>

          <section id="credits" className="scroll-mt-24 space-y-4">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Credits
            </h2>
            <div className="space-y-6 text-muted-foreground">
              <p className="prose max-w-none">
                Professor Emiliano Catan at NYU Law has been an active advisor
                to this project from the beginning. The{" "}
                <a
                  href="https://www.law.nyu.edu/leadershipprogram"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Jacobson Leadership Program in Law and Business (opens in a new tab)"
                >
                  Jacobson Leadership Program in Law and Business
                </a>{" "}
                at NYU has also provided support, including a commitment to
                substantially all of the funding. We are also thankful to NYU
                Law professor Chris Sprigman. The concept and design for this
                site borrow heavily from the{" "}
                <a
                  href="https://case.law"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Caselaw Access Project (opens in a new tab)"
                >
                  Caselaw Access Project
                </a>
                , a product of the{" "}
                <a
                  href="https://lil.law.harvard.edu/"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Library Innovation Lab (opens in a new tab)"
                >
                  Library Innovation Lab
                </a>{" "}
                (LIL) at Harvard Law; Jack Cushman at LIL provided guidance and
                advice early on in this project. Josh Carty provided technical
                assistance early on, and helped brainstorm.
              </p>
              <p className="prose max-w-none">
                This project would not have gotten off the ground without the
                prior work of Peter Adelson, Matthew Jennejohn, Julian Nyarko,
                and Eric Talley, whose{" "}
                <a
                  href="https://onlinelibrary.wiley.com/doi/abs/10.1111/jels.12410"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Introducing a New Corpus of Definitive M&A Agreements, 2000–2020 (opens in a new tab)"
                >
                  <em>
                    Introducing a New Corpus of Definitive M&A Agreements,
                    2000–2020
                  </em>
                </a>{" "}
                provided the inspiration and seed set of source links to
                definitive agreements in EDGAR. Their GitHub repository is
                available{" "}
                <a
                  href="https://github.com/padelson/dma_corpus/tree/main"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="here (opens in a new tab)"
                >
                  here
                </a>
                .
              </p>
            </div>
          </section>
        </div>
      </div>
    </PageShell>
  );
}
