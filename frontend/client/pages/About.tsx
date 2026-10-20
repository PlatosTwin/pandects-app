import {
  ArrowRight,
  BookOpen,
  CalendarClock,
  Cpu,
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
import type { LucideIcon } from "lucide-react";
import type { SVGProps } from "react";
import { FactCard } from "@/components/FactCard";
import { CodeBlock } from "@/components/ui/code-block";
import { ExternalLink } from "@/components/ui/external-link";
import { Eyebrow, eyebrowClassName } from "@/components/ui/eyebrow";
import { PageShell } from "@/components/PageShell";
import { SectionHeader } from "@/components/SectionHeader";
import { Skeleton } from "@/components/ui/skeleton";
import { useAgreementSummary } from "@/hooks/use-agreement-summary";
import { formatDateValue } from "@/lib/format-utils";
import { getDocsUrl } from "@/lib/docs-url";
import { Link } from "react-router-dom";

const docsUrl = `${getDocsUrl()}/docs/guides/getting-started`;
const ssrnUrl =
  "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6653778";
const pandectsBibtex = String.raw`@article{Bogdanov2026Pandects,
  author  = {Bogdanov, Nikita},
  title   = {Pandects: An Open-Source Database of Publicly Filed M\&A Agreements},
  journal = {SSRN Electronic Journal},
  year    = {2026},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6653778}
}`;

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
    // Custom glyph; structurally compatible with a lucide icon component.
    icon: A11ySymbol as unknown as LucideIcon,
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
  icon: LucideIcon;
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
    <PageShell layout="docs" title="About">
      <div className="w-full">
        <article>
          <section
            id="overview"
            className="scroll-mt-24"
            aria-labelledby="overview-heading"
          >
            <SectionHeader title="Overview" id="overview-heading" />

            <div className="mt-6 grid gap-3 sm:grid-cols-2">
              {quickFacts.map(({ icon, eyebrow, value }) => (
                <FactCard
                  key={eyebrow}
                  icon={icon}
                  eyebrow={eyebrow}
                  title={eyebrow === "Coverage" ? coverageValue : value}
                  titleAs="h3"
                />
              ))}
            </div>

            <div className="mt-4 rounded-xl border border-border bg-card p-5">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <Eyebrow>Agreement Index</Eyebrow>
                  <p className="mt-1 text-sm prose-copy">
                    Browse the corpus by year, party, and processing status.
                  </p>
                </div>
                <Link
                  to="/agreement-index"
                  className="inline-flex w-fit shrink-0 items-center gap-2 rounded-lg border border-primary/30 bg-primary/10 px-3 py-2 text-sm font-medium text-primary transition-colors hover:border-primary/50 hover:bg-primary/15"
                >
                  Browse the index
                  <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </Link>
              </div>

              {!summaryError && (
                <div
                  className="mt-4 grid gap-3 sm:grid-cols-3"
                  aria-busy={summaryLoading}
                  aria-live="polite"
                >
                  {corpusMetrics.map(({ key, icon: Icon, label, detail }) => (
                    <div
                      key={key}
                      className="flex min-h-[5.5rem] items-center gap-3 rounded-lg border border-border bg-background p-3"
                    >
                      <div className="flex h-9 w-9 flex-none items-center justify-center rounded-md bg-primary/10 text-primary">
                        <Icon className="h-4 w-4" aria-hidden="true" />
                      </div>
                      <div className="min-w-0">
                        <Eyebrow>{label}</Eyebrow>
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
              <ExternalLink
                href="https://github.com/PlatosTwin/pandects-app/tree/main/examples"
                className="underline underline-offset-2"
              >
                Jupyter Notebook examples
              </ExternalLink>{" "}
              as well as built out a{" "}
              <ExternalLink
                href={docsUrl}
                className="underline underline-offset-2"
              >
                documentation site
              </ExternalLink>
              . Finally, for more control, users may elect to{" "}
              <Link to="/bulk-data" className="underline underline-offset-2">
                bulk download
              </Link>{" "}
              a copy of the complete database.
            </p>
          </section>

          <section
            id="contributing"
            className="scroll-mt-24 pt-12"
            aria-labelledby="contributing-heading"
          >
            <SectionHeader
              title="Contributing"
              id="contributing-heading"
            />

            <p className="prose-copy mt-6">
              This is an open-source project, and contributions are welcome.
              See the{" "}
              <ExternalLink
                href="https://github.com/PlatosTwin/pandects-app"
                className="underline underline-offset-2"
              >
                GitHub repository
              </ExternalLink>{" "}
              for details.
            </p>

            <Eyebrow asChild className="mt-8">
              <h3>Where help is most welcome</h3>
            </Eyebrow>
            <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
              {helpAreas.map(({ icon, title, description }) => (
                <FactCard
                  key={title}
                  icon={icon}
                  title={title}
                  description={description}
                  titleAs="h4"
                />
              ))}
            </div>

            <Eyebrow asChild className="mt-8">
              <h3>On the roadmap</h3>
            </Eyebrow>
            <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
              {roadmapItems.map(({ icon, title, description }) => (
                <FactCard
                  key={title}
                  icon={icon}
                  title={title}
                  description={description}
                  titleAs="h4"
                />
              ))}
            </div>
          </section>

          <section
            id="credits"
            className="scroll-mt-24 pt-12"
            aria-labelledby="credits-heading"
          >
            <SectionHeader title="Credits" id="credits-heading" />

            <dl className="mt-6 grid grid-cols-1 sm:grid-cols-[160px_1fr] gap-x-6 gap-y-5">
              <dt className={`${eyebrowClassName} self-start`}>
                Advisors
              </dt>
              <dd className="text-sm text-foreground">
                Emiliano Catan (NYU Law); Alex Walker (NYU Law, tax taxonomy);
                Chris Sprigman (NYU Law).
              </dd>

              <dt className={`${eyebrowClassName} self-start`}>
                Institutional support
              </dt>
              <dd className="text-sm text-foreground">
                <ExternalLink
                  href="https://www.law.nyu.edu/leadershipprogram"
                  className="underline underline-offset-2"
                >
                  Jacobson Leadership Program in Law and Business
                </ExternalLink>
                ; NYU IT High Performance Computing.
              </dd>

              <dt className={`${eyebrowClassName} self-start`}>
                Inspiration
              </dt>
              <dd className="text-sm text-foreground">
                Site concept and design borrow from{" "}
                <ExternalLink
                  href="https://case.law"
                  className="underline underline-offset-2"
                >
                  Caselaw Access Project
                </ExternalLink>
                , a product of the{" "}
                <ExternalLink
                  href="https://lil.law.harvard.edu/"
                  className="underline underline-offset-2"
                >
                  Library Innovation Lab
                </ExternalLink>{" "}
                (LIL) at Harvard Law. Jack Cushman at LIL provided early
                guidance.
              </dd>

              <dt className={`${eyebrowClassName} self-start`}>
                Prior work
              </dt>
              <dd className="text-sm text-foreground">
                Peter Adelson, Matthew Jennejohn, Julian Nyarko, Eric Talley —{" "}
                <ExternalLink
                  href="https://onlinelibrary.wiley.com/doi/abs/10.1111/jels.12410"
                  className="underline underline-offset-2"
                >
                  <em>
                    Introducing a New Corpus of Definitive M&A Agreements,
                    2000–2020
                  </em>
                </ExternalLink>
                .{" "}
                <ExternalLink
                  href="https://github.com/padelson/dma_corpus/tree/main"
                  className="underline underline-offset-2"
                >
                  DMA corpus repository
                </ExternalLink>
                .
              </dd>

              <dt className={`${eyebrowClassName} self-start`}>
                Technical assistance
              </dt>
              <dd className="text-sm text-foreground">
                Josh Carty (early help and brainstorming).
              </dd>
            </dl>
          </section>

          <section
            id="cite-pandects"
            className="scroll-mt-24 pt-12"
            aria-labelledby="cite-pandects-heading"
          >
            <SectionHeader
              title="Cite Pandects"
              id="cite-pandects-heading"
            />

            <p className="prose-copy mt-6">
              If you use Pandects in academic work, please cite:
            </p>
            <p className="prose-copy mt-3">
              Bogdanov, Nikita.{" "}
              <em>
                Pandects: An Open-Source Database of Publicly Filed M&amp;A
                Agreements
              </em>
              . 2026. Available at SSRN:{" "}
              <ExternalLink
                href={ssrnUrl}
                className="underline underline-offset-2"
              >
                {ssrnUrl}
              </ExternalLink>
              .
            </p>
            <CodeBlock
              className="mt-4"
              code={pandectsBibtex}
              label="BibTeX"
              copyLabel="Copy BibTeX citation"
            />
          </section>
        </article>
      </div>
    </PageShell>
  );
}
