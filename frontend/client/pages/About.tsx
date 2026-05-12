import {
  ArrowRight,
  BookOpen,
  CalendarClock,
  Cpu,
  ExternalLink,
  FileCode2,
  FileText,
  LineChart,
  Layers,
  ListTree,
  Plug,
  Search,
  Shield,
  Tags,
} from "lucide-react";
import type { SVGProps } from "react";
import { PageShell } from "@/components/PageShell";
import { Skeleton } from "@/components/ui/skeleton";
import { useAgreementSummary } from "@/hooks/use-agreement-summary";
import { formatDateValue } from "@/lib/format-utils";
import { Link } from "react-router-dom";

const docsUrl = "https://docs.pandects.org/docs/guides/getting-started";

function A11ySymbol(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="7" r="1" />
      <path d="M7.5 10h9" />
      <path d="M12 10v3.5" />
      <path d="m9.5 18 2.5-4.5 2.5 4.5" />
    </svg>
  );
}

const quickFacts = [
  {
    icon: FileText,
    eyebrow: "Coverage",
    value: "Definitive merger agreements",
  },
  {
    icon: FileCode2,
    eyebrow: "Structure",
    value: "XML",
  },
  {
    icon: Plug,
    eyebrow: "Access",
    value: "UI Search, API, MCP, bulk",
  },
  {
    icon: CalendarClock,
    eyebrow: "Cadence",
    value: "Monthly refresh",
  },
];

const corpusMetrics = [
  {
    key: "agreements",
    icon: FileText,
    label: "Agreements",
    detail: "Ingested",
  },
  {
    key: "sections",
    icon: Layers,
    label: "Sections",
    detail: "Indexed",
  },
  {
    key: "pages",
    icon: BookOpen,
    label: "Pages",
    detail: "Parsed",
  },
] as const;

const helpAreas = [
  {
    icon: Shield,
    title: "Security hardening",
    description: "Including OAuth flow reviews.",
  },
  {
    icon: A11ySymbol,
    title: "Accessibility",
    description: "Keyboard, focus, contrast.",
  },
  {
    icon: Cpu,
    title: "ML model optimization",
    description: "Performance and quality.",
  },
];

const roadmapItems: {
  icon: typeof Tags;
  title: string;
  description?: string;
}[] = [
  {
    icon: Tags,
    title: "Taxonomize clauses broadly",
    description:
      "Not just tax — important clauses generally. Today we taxonomize sections, but a single section can contain multiple important clauses.",
  },
  {
    icon: Search,
    title: "Defined terms",
    description:
      "Identify them (including those in appendices) and make them searchable.",
  },
  {
    icon: LineChart,
    title: "'Is this market?'",
    description:
      "A specific UI surface to help practitioners answer market questions.",
  },
  {
    icon: ListTree,
    title: "Refine the section taxonomy",
  },
];

const extIcon = (
  <ExternalLink
    className="ml-0.5 inline-block h-3 w-3 align-baseline opacity-60"
    aria-hidden="true"
  />
);

export default function About() {
  const {
    summary,
    isLoading: summaryLoading,
    error: summaryError,
  } = useAgreementSummary();
  const latestFilingDate =
    summary?.latest_filing_date && !summaryError
      ? formatDateValue(summary.latest_filing_date)
      : null;
  const coverageValue = latestFilingDate
    ? `Jan. 1, 2000 - ${latestFilingDate}`
    : "Since Jan. 1, 2000";

  return (
    <PageShell size="lg" title="About">
      <div className="w-full">
        <article>
          <section
            id="overview"
            className="scroll-mt-24 first:pt-0 pt-12"
            aria-labelledby="overview-heading"
          >
            <div className="flex items-center gap-3">
              <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 text-primary text-xs font-medium tabular-nums">
                01
              </span>
              <h2
                id="overview-heading"
                className="text-2xl font-semibold tracking-tight text-foreground"
              >
                Overview
              </h2>
            </div>

            <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              {quickFacts.map(({ icon: Icon, eyebrow, value }) => (
                <div
                  key={eyebrow}
                  className="rounded-lg border border-border bg-card p-4"
                >
                  <Icon className="h-5 w-5 text-primary mb-2" />
                  <div className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                    {eyebrow}
                  </div>
                  <div className="mt-1 text-sm font-medium leading-5 text-foreground">
                    {eyebrow === "Coverage" ? coverageValue : value}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 rounded-lg border border-border bg-card p-4 shadow-sm sm:p-5">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div className="max-w-2xl">
                  <div className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                    Agreement Index
                  </div>
                  <p className="mt-1 text-sm prose-copy">
                    Browse the corpus by year, party, and processing status.
                  </p>
                </div>
                <Link
                  to="/agreement-index"
                  className="inline-flex w-fit items-center gap-2 rounded-md border border-primary/30 bg-primary/10 px-3 py-2 text-sm font-medium text-primary transition-colors hover:border-primary/50 hover:bg-primary/15"
                >
                  Browse the index
                  <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </Link>
              </div>

              {!summaryError && (
                <div
                  className="mt-4 grid gap-3 md:grid-cols-3"
                  aria-busy={summaryLoading}
                  aria-live="polite"
                >
                  {corpusMetrics.map(({ key, icon: Icon, label, detail }) => (
                    <div
                      key={key}
                      className="flex min-h-[5.5rem] items-center gap-3 rounded-md border border-border bg-background/80 p-3"
                    >
                      <div className="flex h-9 w-9 flex-none items-center justify-center rounded-md bg-primary/10 text-primary">
                        <Icon className="h-4 w-4" aria-hidden="true" />
                      </div>
                      <div className="min-w-0">
                        <div className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                          {label}
                        </div>
                        <div className="flex min-h-7 items-center text-xl font-semibold tabular-nums text-foreground">
                          {summaryLoading ? (
                            <>
                              <Skeleton className="h-6 w-20" />
                              <span className="sr-only">Loading {label}</span>
                            </>
                          ) : (
                            (summary?.[key] ?? 0).toLocaleString("en-US")
                          )}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {detail}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <p className="prose-copy mt-6">
              Pandects is an <strong>open-source M&A research platform</strong>{" "}
              built to make it easier to browse and analyze sections across
              definitive merger agreements. Unlike other corpora, we update our
              database on a <strong>monthly basis</strong>, and make available
              not just EDGAR URLs or unprocessed HTML, but also{" "}
              <strong>XML</strong>, compiled with purpose-built ML and data
              orchestration pipelines. On top of exposing XML, we{" "}
              <strong>taxonomize</strong> each section of each agreement into a
              comprehensive taxonomy, and enrich each deal with detailed
              metadata.
            </p>
            <p className="prose-copy mt-4">
              While we expose a web-based search interface, the real power of
              the Pandects platform lies in the <strong>API </strong>and
              associated <strong>MCP server</strong>. By creating a free
              account, you gain access to unredacted XML and unlock higher rate
              limits for public endpoints, putting Pandects data at your
              fingertips. An account also provides access to the MCP server,
              enabling you to tackle research questions by partnering with an
              AI research agent of your choice. To aid adoption of both the API
              and MCP server, we've put together a small collection of{" "}
              <a
                href="https://github.com/PlatosTwin/pandects-app/tree/main/examples"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="GitHub directory of API examples (opens in a new tab)"
                className="underline underline-offset-2"
              >
                Jupyter Notebook examples
                {extIcon}
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
                {extIcon}
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
                {extIcon}
              </a>{" "}
              a copy of the complete database.
            </p>
          </section>

          <section
            id="contributing"
            className="scroll-mt-24 pt-12"
            aria-labelledby="contributing-heading"
          >
            <div className="flex items-center gap-3">
              <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 text-primary text-xs font-medium tabular-nums">
                02
              </span>
              <h2
                id="contributing-heading"
                className="text-2xl font-semibold tracking-tight text-foreground"
              >
                Contributing
              </h2>
            </div>

            <p className="prose-copy mt-6">
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
                {extIcon}
              </a>{" "}
              for details.
            </p>

            <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-muted-foreground mt-8">
              Where help is most welcome
            </h3>
            <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
              {helpAreas.map(({ icon: Icon, title, description }) => (
                <div
                  key={title}
                  className="rounded-lg border border-border bg-card p-4"
                >
                  <Icon className="h-5 w-5 text-primary mb-2" />
                  <h4 className="text-base font-medium text-foreground">
                    {title}
                  </h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    {description}
                  </p>
                </div>
              ))}
            </div>

            <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-muted-foreground mt-8">
              On the roadmap
            </h3>
            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
              {roadmapItems.map(({ icon: Icon, title, description }) => (
                <div
                  key={title}
                  className="rounded-lg border border-border bg-card p-4"
                >
                  <Icon className="h-5 w-5 text-primary mb-2" />
                  <h4 className="text-base font-medium text-foreground">
                    {title}
                  </h4>
                  {description && (
                    <p className="text-sm text-muted-foreground mt-1">
                      {description}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </section>

          <section
            id="credits"
            className="scroll-mt-24 pt-12"
            aria-labelledby="credits-heading"
          >
            <div className="flex items-center gap-3">
              <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 text-primary text-xs font-medium tabular-nums">
                03
              </span>
              <h2
                id="credits-heading"
                className="text-2xl font-semibold tracking-tight text-foreground"
              >
                Credits
              </h2>
            </div>

            <dl className="mt-6 grid grid-cols-1 sm:grid-cols-[160px_1fr] gap-x-6 gap-y-5">
              <dt className="text-xs uppercase tracking-[0.12em] text-muted-foreground self-start">
                Advisors
              </dt>
              <dd className="text-sm text-foreground">
                Emiliano Catan (NYU Law); Alex Walker (NYU Law, tax taxonomy);
                Chris Sprigman (NYU Law).
              </dd>

              <dt className="text-xs uppercase tracking-[0.12em] text-muted-foreground self-start">
                Institutional support
              </dt>
              <dd className="text-sm text-foreground">
                <a
                  href="https://www.law.nyu.edu/leadershipprogram"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Jacobson Leadership Program in Law and Business (opens in a new tab)"
                  className="underline underline-offset-2"
                >
                  Jacobson Leadership Program in Law and Business
                  {extIcon}
                </a>
                ; NYU IT High Performance Computing.
              </dd>

              <dt className="text-xs uppercase tracking-[0.12em] text-muted-foreground self-start">
                Inspiration
              </dt>
              <dd className="text-sm text-foreground">
                Site concept and design borrow from{" "}
                <a
                  href="https://case.law"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Caselaw Access Project (opens in a new tab)"
                  className="underline underline-offset-2"
                >
                  Caselaw Access Project
                  {extIcon}
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
                  {extIcon}
                </a>{" "}
                (LIL) at Harvard Law. Jack Cushman at LIL provided early
                guidance.
              </dd>

              <dt className="text-xs uppercase tracking-[0.12em] text-muted-foreground self-start">
                Prior work
              </dt>
              <dd className="text-sm text-foreground">
                Peter Adelson, Matthew Jennejohn, Julian Nyarko, Eric Talley —{" "}
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
                  {extIcon}
                </a>
                .{" "}
                <a
                  href="https://github.com/padelson/dma_corpus/tree/main"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="DMA corpus repository (opens in a new tab)"
                  className="underline underline-offset-2"
                >
                  DMA corpus repository
                  {extIcon}
                </a>
                .
              </dd>

              <dt className="text-xs uppercase tracking-[0.12em] text-muted-foreground self-start">
                Technical assistance
              </dt>
              <dd className="text-sm text-foreground">
                Josh Carty (early help and brainstorming).
              </dd>
            </dl>
          </section>
        </article>
      </div>
    </PageShell>
  );
}
