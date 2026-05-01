import { PageShell } from "@/components/PageShell";

export default function About() {
  const docsUrl = "https://docs.pandects.org/docs/guides/getting-started";

  return (
    <PageShell size="xl" title="About">
      <article className="space-y-12">
        <section
          id="overview"
          className="scroll-mt-24 space-y-4 border-t border-border pt-6"
          aria-labelledby="overview-heading"
        >
          <h2
            id="overview-heading"
            className="text-2xl font-semibold tracking-tight text-foreground"
          >
            Overview
          </h2>
          <p className="prose prose-copy max-w-none">
            Pandects is an <strong>open-source M&A research platform</strong>{" "}
            built to make it easier to browse and analyze sections across
            definitive merger agreements. Unlike other corpora, we update our
            database on a <strong>monthly basis</strong>, and make available not
            just EDGAR URLs or unprocessed HTML, but also <strong>XML</strong>,
            compiled with purpose-built ML and data orchestration pipelines. On
            top of exposing XML, we <strong>taxonomize</strong> each section of
            each agreement into a comprehensive taxonomy, and enrich each deal
            with detailed metadata.
          </p>
          <p className="prose prose-copy max-w-none">
            While we expose a web-based search interface, the real power of the
            Pandects platform lies in the <strong>API </strong>and associated{" "}
            <strong>MCP server</strong>. By creating a free account, you gain
            access to unredacted XML and unlock higher rate limits for public
            endpoints, putting Pandects data at your fingertips. An account also
            provides access to the MCP server, enabling you to tackle research
            questions by partnering with an AI research agent of your choice. To
            aid adoption of both the API and MCP server, we've put together a
            small collection of{" "}
            <a
              href="https://github.com/PlatosTwin/pandects-app/tree/main/examples"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub directory of API examples (opens in a new tab)"
              className="underline underline-offset-2"
            >
              Jupyter Notebook examples
            </a>{" "}
            as well as built out a{" "}
            <a
              href={docsUrl}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Documentation site (opens in a new tab)"
              className="underline underline-offset-2"
            >
              documentation site
            </a>
            . Finally, for more control, users may elect to{" "}
            <a
              href="https://pandects.org/bulk-data/"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Bulk data download page (opens in a new tab)"
              className="underline underline-offset-2"
            >
              bulk download
            </a>{" "}
            a copy of the complete database.
          </p>
        </section>

        <section
          id="contributing"
          className="scroll-mt-24 space-y-4 border-t border-border pt-6"
          aria-labelledby="contributing-heading"
        >
          <h2
            id="contributing-heading"
            className="text-2xl font-semibold tracking-tight text-foreground"
          >
            Contributing
          </h2>
          <p className="prose prose-copy max-w-none">
            This is an open-source project, and contributions are welcome. See
            the{" "}
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
          <p className="prose-copy">
            We are especially thankful for focused contributions to:
          </p>
          <ul className="prose-copy list-disc space-y-2 pl-6">
            <li>Security hardening, including OAuth flow reviews.</li>
            <li>Accessibility improvements (keyboard, focus, contrast).</li>
            <li>ML model optimization.</li>
          </ul>
          <p className="prose-copy">
            In the medium term, we hope to:
          </p>
          <ul className="prose-copy list-disc space-y-2 pl-6">
            <li>
              Taxonomize not just <em>tax</em> clauses but also important
              clauses <em>generally</em>. Right now, in addition to tax clauses,
              we taxonomize <em>sections</em>, but a single section can have
              multiple important clauses within it.
            </li>
            <li>
              Identify defined terms, including those relegated to appendices,
              and make them searchable.
            </li>
            <li>
              Develop a specific UI surface to help practitioners answer, "Is
              this market?" or "What is market?" questions.
            </li>
            <li>Refine the section taxonomy.</li>
          </ul>
        </section>

        <section
          id="credits"
          className="scroll-mt-24 space-y-4 border-t border-border pt-6"
          aria-labelledby="credits-heading"
        >
          <h2
            id="credits-heading"
            className="text-2xl font-semibold tracking-tight text-foreground"
          >
            Credits
          </h2>
          <div className="space-y-6 text-muted-foreground">
            <p className="prose prose-copy max-w-none">
              Professor Emiliano Catan at NYU Law has been an active advisor to
              this project from the beginning. The{" "}
              <a
                href="https://www.law.nyu.edu/leadershipprogram"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Jacobson Leadership Program in Law and Business (opens in a new tab)"
                className="underline underline-offset-2"
              >
                Jacobson Leadership Program in Law and Business
              </a>{" "}
              at NYU has also provided support, including a commitment to all of
              the startup funding. NYU Law professor Alex Walker developed the
              tax taxonomy, for which we are very thankful. We are also thankful
              to NYU Law professor Chris Sprigman. The concept and design for
              this site borrow heavily from the{" "}
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
            <p className="prose prose-copy max-w-none">
              This project would not have gotten off the ground without the
              prior work of Peter Adelson, Matthew Jennejohn, Julian Nyarko, and
              Eric Talley, whose{" "}
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
            <p className="prose prose-copy max-w-none">
              Finally, this project was supported in part through NYU IT High
              Performance Computing resources, services, and staff expertise.
            </p>
          </div>
        </section>
      </article>
    </PageShell>
  );
}
