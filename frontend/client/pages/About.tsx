import { PageShell } from "@/components/PageShell";

export default function About() {
  return (
    <PageShell
      size="xl"
      title="About"
    >
      <article className="space-y-12">
        <section id="overview" className="scroll-mt-24 space-y-4 border-t border-border/60 pt-6" aria-labelledby="overview-heading">
          <h2 id="overview-heading" className="text-2xl font-semibold tracking-tight text-foreground">
            Overview
          </h2>
          <p className="text-muted-foreground">
            Pandects is an open-source M&A research platform built to make it
            easier to browse and analyze clauses across definitive agreements.
          </p>
          <p className="text-muted-foreground">
            We plan to soft-launch the site in mid to late March 2026. Check back then
            for more complete data.
          </p>
        </section>

        <section id="contributing" className="scroll-mt-24 space-y-4 border-t border-border/60 pt-6" aria-labelledby="contributing-heading">
          <h2 id="contributing-heading" className="text-2xl font-semibold tracking-tight text-foreground">
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
              className="underline underline-offset-2"
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

        <section id="credits" className="scroll-mt-24 space-y-4 border-t border-border/60 pt-6" aria-labelledby="credits-heading">
          <h2 id="credits-heading" className="text-2xl font-semibold tracking-tight text-foreground">
            Credits
          </h2>
          <div className="space-y-6 text-muted-foreground">
            <p className="prose max-w-none text-muted-foreground">
              Professor Emiliano Catan at NYU Law has been an active advisor
              to this project from the beginning. The{" "}
              <a
                href="https://www.law.nyu.edu/leadershipprogram"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Jacobson Leadership Program in Law and Business (opens in a new tab)"
                className="underline underline-offset-2"
              >
                Jacobson Leadership Program in Law and Business
              </a>{" "}
              at NYU has also provided support, including a commitment to
              substantially all of the startup funding. We are also thankful to NYU
              Law professor Chris Sprigman. The concept and design for this
              site borrow heavily from the{" "}
              <a
                href="https://case.law"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Caselaw Access Project (opens in a new tab)"
                className="underline underline-offset-2"
              >
                Caselaw Access Project
              </a>
              , a product of the{" "}
              <a
                href="https://lil.law.harvard.edu/"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Library Innovation Lab (opens in a new tab)"
                className="underline underline-offset-2"
              >
                Library Innovation Lab
              </a>{" "}
              (LIL) at Harvard Law; Jack Cushman at LIL provided guidance and
              advice early on in this project. Josh Carty provided technical
              assistance early on, and helped brainstorm.
            </p>
            <p className="prose max-w-none text-muted-foreground">
              This project would not have gotten off the ground without the
              prior work of Peter Adelson, Matthew Jennejohn, Julian Nyarko,
              and Eric Talley, whose{" "}
              <a
                href="https://onlinelibrary.wiley.com/doi/abs/10.1111/jels.12410"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Introducing a New Corpus of Definitive M&A Agreements, 2000–2020 (opens in a new tab)"
                className="underline underline-offset-2"
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
                aria-label="DMA corpus repository (opens in a new tab)"
                className="underline underline-offset-2"
              >
                DMA corpus repository
              </a>
              .
            </p>
            <p className="prose max-w-none text-muted-foreground">
              Finally, this project was supported in part through the NYU IT High Performance Computing resources, services, and staff expertise.
            </p>
          </div>
        </section>
      </article>
    </PageShell>
  );
}
