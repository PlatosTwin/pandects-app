import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";

export default function SourcesMethods() {
  const [activeSection, setActiveSection] = useState("");
  const firstStepRef = useRef<HTMLDivElement | null>(null);
  const lastStepRef = useRef<HTMLDivElement | null>(null);
  const pipelineRef = useRef<HTMLDivElement | null>(null);
  const [pipelineProgress, setPipelineProgress] = useState(0);
  const [pipelineLine, setPipelineLine] = useState({ top: 0, height: 0 });
  const [isCoarsePointer, setIsCoarsePointer] = useState(false);

  const navItems = useMemo(
    () => [
      { id: "overview", label: "Overview" },
      { id: "data-pipeline-architecture", label: "Data Pipeline Architecture" },
      { id: "ml-models", label: "ML Models" },
      {
        id: "agreement-identification",
        label: "Agreement Model",
        indent: true,
      },
      {
        id: "page-classification",
        label: "Page Classifier Model",
        indent: true,
      },
      { id: "page-tagging", label: "Tagging Model", indent: true },
      {
        id: "section-classification",
        label: "Taxonomy Model",
        indent: true,
      },
      { id: "gaps-and-callouts", label: "Gaps and Other Call Outs" },
      { id: "validations", label: "Validations" },
    ],
    []
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

    const timer = window.setTimeout(syncFromHash, 0);

    window.addEventListener("hashchange", syncFromHash);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("hashchange", syncFromHash);
    };
  }, []);

  useEffect(() => {
    const updateProgress = () => {
      const firstEl = firstStepRef.current;
      const lastEl = lastStepRef.current;
      if (!firstEl || !lastEl) return;
      const firstRect = firstEl.getBoundingClientRect();
      const lastRect = lastEl.getBoundingClientRect();
      if (!firstRect.height || !lastRect.height) return;
      const containerRect = pipelineRef.current?.getBoundingClientRect();
      if (containerRect) {
        const top = firstRect.top - containerRect.top + firstRect.height / 2;
        const bottom = lastRect.top - containerRect.top + lastRect.height / 2;
        const height = Math.max(0, bottom - top);
        setPipelineLine((prev) =>
          Math.abs(prev.top - top) > 0.5 || Math.abs(prev.height - height) > 0.5
            ? { top, height }
            : prev
        );
      }
      const viewportHeight = window.innerHeight || 1;
      const viewportMid = window.scrollY + viewportHeight / 2;
      const firstCenter = firstRect.top + window.scrollY + firstRect.height / 2;
      const lastCenter = lastRect.top + window.scrollY + lastRect.height / 2;
      const rawProgress =
        (viewportMid - firstCenter) / (lastCenter - firstCenter);
      const clamped = Math.min(1, Math.max(0, rawProgress));
      setPipelineProgress(clamped);
    };

    let raf = 0;
    const onScroll = () => {
      if (raf) return;
      raf = window.requestAnimationFrame(() => {
        raf = 0;
        updateProgress();
      });
    };

    updateProgress();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) window.cancelAnimationFrame(raf);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(pointer: coarse)");
    const updatePointer = () => setIsCoarsePointer(media.matches);
    updatePointer();
    if (media.addEventListener) {
      media.addEventListener("change", updatePointer);
      return () => media.removeEventListener("change", updatePointer);
    }
    media.addListener(updatePointer);
    return () => media.removeListener(updatePointer);
  }, []);

  const ComingSoon = ({ title }: { title: string }) => (
    <Card className="border-border/70 bg-card/70 p-5">
      <div className="text-sm font-medium text-foreground">{title}</div>
      <p className="mt-1 text-sm text-muted-foreground">
        This section is being written. If you would like to help, open an issue
        with suggestions.
      </p>
    </Card>
  );

  const classifierLabels = [
    "front_matter",
    "toc",
    "body",
    "sig",
    "back_matter",
  ];
  const classifierAbbreviations = ["FM", "TOC", "BDY", "SIG", "BM"];
  const pageClassifierMatrix = [
    [25, 0, 0, 0, 0],
    [0, 94, 2, 0, 0],
    [0, 2, 1506, 3, 35],
    [0, 0, 0, 32, 1],
    [0, 0, 19, 3, 174],
  ];
  const pageClassifierMetrics = [
    { label: "front_matter", acc: 1.0, p: 1.0, r: 1.0, f1: 1.0 },
    { label: "toc", acc: 0.9792, p: 0.9792, r: 0.9792, f1: 0.9792 },
    { label: "body", acc: 0.9741, p: 0.9862, r: 0.9741, f1: 0.9801 },
    { label: "sig", acc: 0.9697, p: 0.8421, r: 0.9697, f1: 0.9014 },
    {
      label: "back_matter",
      acc: 0.8878,
      p: 0.8286,
      r: 0.8878,
      f1: 0.8571,
    },
  ];
  const finalClassifierLabels = classifierLabels;
  const finalClassifierAbbreviations = classifierAbbreviations;
  const finalClassifierMatrix = [
    [31, 0, 0, 0, 0],
    [0, 115, 0, 0, 0],
    [0, 0, 1879, 0, 0],
    [0, 0, 0, 38, 1],
    [0, 0, 0, 0, 195],
  ];
  const finalClassifierMetrics = [
    { label: "front_matter", acc: 1.0, p: 1.0, r: 1.0, f1: 1.0 },
    { label: "toc", acc: 1.0, p: 1.0, r: 1.0, f1: 1.0 },
    { label: "body", acc: 1.0, p: 1.0, r: 1.0, f1: 1.0 },
    { label: "sig", acc: 0.9744, p: 1.0, r: 0.9744, f1: 0.987 },
    {
      label: "back_matter",
      acc: 1.0,
      p: 0.9949,
      r: 1.0,
      f1: 0.9974,
    },
  ];
  const formatMetric = (value: number) => `${(value * 100).toFixed(2)}%`;

  return (
    <PageShell
      size="xl"
      title="Sources & Methods"
      subtitle="Where the data comes from, and how we turn text and HTML into XML."
    >
      <div className="grid gap-8 lg:grid-cols-[280px_1fr]">
        <aside className="hidden lg:block">
          <div className="sticky top-20">
            <Card className="border-border/70 bg-background/70 p-3 backdrop-blur">
              <div className="px-2 pb-2 pt-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                On this page
              </div>
              <nav aria-label="Sources and methods page sections">
                <ul className="space-y-1">
                  {navItems.map(({ id, label, indent }) => (
                    <li key={id} className={indent ? "ml-4" : undefined}>
                      <button
                        type="button"
                        onClick={() => scrollToSection(id)}
                        aria-current={
                          activeSection === id ? "location" : undefined
                        }
                        aria-controls={id}
                        className={cn(
                          "w-full rounded-md px-3 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                          activeSection === id
                            ? "bg-primary/10 text-primary font-medium"
                            : "text-muted-foreground hover:bg-accent hover:text-foreground"
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
            <h2 className="sr-only">Overview</h2>
            <p className="max-w-3xl text-muted-foreground">
              Pandects sources agreements from the SEC's EDGAR database, then
              runs each agreement through a purpose-built pipeline that turns
              text and messy HTML into clean XML. Conceptually, our pipeline
              solves five distinct problems.
            </p>
            <div className="max-w-4xl rounded-2xl border border-border/70 bg-card/50 p-5 shadow-sm">
              <div className="grid gap-1 border-b border-border/60 px-4 pb-4 text-xs font-semibold uppercase tracking-wide text-muted-foreground md:grid-cols-[16px_1fr_1fr]">
                <div aria-hidden="true" />
                <div className="px-1 text-left">The Status Quo</div>
                <div className="px-2 text-left">The Pandects Standard</div>
              </div>
              <ol
                className="space-y-4 pt-4"
                aria-label="Status quo versus Pandects standard comparisons"
              >
                <li className="rounded-xl border border-border/60 bg-background/60 p-4">
                  <div className="grid gap-2 md:grid-cols-[16px_1fr_1fr] md:gap-x-2 md:gap-y-0">
                    <div
                      aria-hidden="true"
                      className="text-sm font-semibold text-foreground md:pt-3"
                    >
                      1
                    </div>
                    <div className="grid content-start gap-2 rounded-lg px-1 py-3 md:col-start-2 md:row-span-2">
                      <span className="sr-only">Status quo:</span>
                      <div className="text-sm font-semibold text-foreground">
                        No Scalable Filtering
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Finding definitive M&A agreements in EDGAR is like
                        finding a needle in a haystack.
                      </p>
                    </div>
                    <div className="grid content-start gap-2 rounded-lg bg-emerald-500/10 px-2 py-3 md:col-start-3 md:row-span-2">
                      <span className="sr-only">Pandects standard:</span>
                      <div className="text-sm font-semibold text-foreground">
                        Automated Identification
                      </div>
                      <p className="text-sm text-foreground">
                        We anchor our search with the{" "}
                        <a
                          href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4731282"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary hover:underline"
                        >
                          DMA Corpus
                        </a>{" "}
                        and develop purpose-build models to identify definitive
                        agreements natively.
                      </p>
                    </div>
                  </div>
                </li>
                <li className="rounded-xl border border-border/60 bg-background/60 p-4">
                  <div className="grid gap-2 md:grid-cols-[16px_1fr_1fr] md:gap-x-2 md:gap-y-0">
                    <div
                      aria-hidden="true"
                      className="text-sm font-semibold text-foreground md:pt-3"
                    >
                      2
                    </div>
                    <div className="grid content-start gap-2 rounded-lg px-1 py-3 md:col-start-2 md:row-span-2">
                      <span className="sr-only">Status quo:</span>
                      <div className="text-sm font-semibold text-foreground">
                        Messy HTML
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Existing datasets provide only links to raw SEC filings,
                        forcing researchers to build their own parsers to
                        extract text and identify articles and sections.
                      </p>
                    </div>
                    <div className="grid content-start gap-2 rounded-lg bg-emerald-500/10 px-2 py-3 md:col-start-3 md:row-span-2">
                      <span className="sr-only">Pandects standard:</span>
                      <div className="text-sm font-semibold text-foreground">
                        Clean, Normalized XML
                      </div>
                      <p className="text-sm text-foreground">
                        We normalize every agreement into standardized XML,
                        stripping away the HTML mess so researchers don't have
                        to invent their own parsing logic.
                      </p>
                    </div>
                  </div>
                </li>
                <li className="rounded-xl border border-border/60 bg-background/60 p-4">
                  <div className="grid gap-2 md:grid-cols-[16px_1fr_1fr] md:gap-x-2 md:gap-y-0">
                    <div
                      aria-hidden="true"
                      className="text-sm font-semibold text-foreground md:pt-3"
                    >
                      3
                    </div>
                    <div className="grid content-start gap-2 rounded-lg px-1 py-3 md:col-start-2 md:row-span-2">
                      <span className="sr-only">Status quo:</span>
                      <div className="text-sm font-semibold text-foreground">
                        A Sea of Clauses
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Whether working with raw or structured text, it's
                        challenging to classify clauses at scale.
                      </p>
                    </div>
                    <div className="grid content-start gap-2 rounded-lg bg-emerald-500/10 px-2 py-3 md:col-start-3 md:row-span-2">
                      <span className="sr-only">Pandects standard:</span>
                      <div className="text-sm font-semibold text-foreground">
                        Rigorous Taxonomy
                      </div>
                      <p className="text-sm text-foreground">
                        We classify every section into a specific clause
                        taxonomy, making it effortless to query specific clause
                        types across thousands of deals.
                      </p>
                    </div>
                  </div>
                </li>
                <li className="rounded-xl border border-border/60 bg-background/60 p-4">
                  <div className="grid gap-2 md:grid-cols-[16px_1fr_1fr] md:gap-x-2 md:gap-y-0">
                    <div
                      aria-hidden="true"
                      className="text-sm font-semibold text-foreground md:pt-3"
                    >
                      4
                    </div>
                    <div className="grid content-start gap-2 rounded-lg px-1 py-3 md:col-start-2 md:row-span-2">
                      <span className="sr-only">Status quo:</span>
                      <div className="text-sm font-semibold text-foreground">
                        Missing Deal Context
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Structured text is of limited use without metadata on
                        deal value, industry codes, and buyer types.
                      </p>
                    </div>
                    <div className="grid content-start gap-2 rounded-lg bg-emerald-500/10 px-2 py-3 md:col-start-3 md:row-span-2">
                      <span className="sr-only">Pandects standard:</span>
                      <div className="text-sm font-semibold text-foreground">
                        LLM-Enriched Data
                      </div>
                      <p className="text-sm text-foreground">
                        We commission LLMs to inject deal-specific
                        context—including consideration amounts, NAICS codes,
                        and PE involvement—directly into the dataset.
                      </p>
                    </div>
                  </div>
                </li>
                <li className="rounded-xl border border-border/60 bg-background/60 p-4">
                  <div className="grid gap-2 md:grid-cols-[16px_1fr_1fr] md:gap-x-2 md:gap-y-0">
                    <div
                      aria-hidden="true"
                      className="text-sm font-semibold text-foreground md:pt-3"
                    >
                      5
                    </div>
                    <div className="grid content-start gap-2 rounded-lg px-1 py-3 md:col-start-2 md:row-span-2">
                      <span className="sr-only">Status quo:</span>
                      <div className="text-sm font-semibold text-foreground">
                        Static Snapshots
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Traditional legal corpora are frozen in time, losing
                        relevance within a few years of publication.
                      </p>
                    </div>
                    <div className="grid content-start gap-2 rounded-lg bg-emerald-500/10 px-2 py-3 md:col-start-3 md:row-span-2">
                      <span className="sr-only">Pandects standard:</span>
                      <div className="text-sm font-semibold text-foreground">
                        Weekly Updates
                      </div>
                      <p className="text-sm text-foreground">
                        Our pipeline runs weekly, ensuring the Pandects dataset
                        remains in sync with the latest market activity.
                      </p>
                    </div>
                  </div>
                </li>
              </ol>
            </div>
          </section>

          <section
            id="data-pipeline-architecture"
            className="scroll-mt-24 space-y-4"
          >
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Pipeline Architecture
            </h2>
            <p className="max-w-3xl text-muted-foreground">
              Our pipeline takes raw EDGAR filings and produces clean XML,
              structured sections, and taxonomy labels. We use{" "}
              <a
                href="https://dagster.io/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                Dagster
              </a>{" "}
              to orchestrate the pipeline, which is defined in full on{" "}
              <a
                href="https://github.com/PlatosTwin/pandects-app/tree/main/etl/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                GitHub
              </a>
              .
            </p>
            <div ref={pipelineRef} className="relative mt-6 max-w-5xl">
              <div
                aria-hidden="true"
                className="pointer-events-none absolute left-4 -translate-x-1/2 w-[2px]"
                style={{ top: pipelineLine.top, height: pipelineLine.height }}
              >
                <div className="absolute left-0 top-0 h-full w-full bg-[linear-gradient(to_bottom,hsl(var(--border)/0.6)_0,hsl(var(--border)/0.6)_6px,transparent_6px,transparent_12px)] bg-[length:2px_12px] bg-repeat-y" />
                <div
                  className="absolute left-0 top-0 h-full w-full origin-top bg-gradient-to-b from-emerald-300 via-emerald-500 to-emerald-600 shadow-[0_0_10px_rgba(16,185,129,0.35)] will-change-transform"
                  style={{ transform: `scaleY(${pipelineProgress})` }}
                />
              </div>
              <ol className="space-y-8" aria-label="Pipeline steps">
                <li className="relative flex gap-4">
                  <div
                    ref={firstStepRef}
                    className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70"
                  >
                    1
                  </div>
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-foreground">
                        Ingestion
                      </span>
                      <a
                        href="#agreement-identification"
                        className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-muted-foreground transition-colors hover:border-emerald-500/40 hover:text-foreground"
                      >
                        Agreement Model
                      </a>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Identifies new definitive agreements and stages links to
                      the raw content on EDGAR for processing.
                    </p>
                  </div>
                </li>
                <li className="relative flex gap-4">
                  <div className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70">
                    2
                  </div>
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-foreground">
                        Normalization
                      </span>
                      <a
                        href="#page-classification"
                        className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-muted-foreground transition-colors hover:border-emerald-500/40 hover:text-foreground"
                      >
                        Page Classifier Model
                      </a>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Splits agreements into pages, sanitizes HTML, and
                      classifies each as{" "}
                      <span className="font-mono text-xs text-foreground">
                        front_matter
                      </span>
                      ,{" "}
                      <span className="font-mono text-xs text-foreground">
                        toc
                      </span>
                      ,{" "}
                      <span className="font-mono text-xs text-foreground">
                        body
                      </span>
                      ,{" "}
                      <span className="font-mono text-xs text-foreground">
                        sig
                      </span>
                      , or{" "}
                      <span className="font-mono text-xs text-foreground">
                        back_matter
                      </span>
                      .
                    </p>
                    <div className="rounded-xl border border-border/60 bg-background/60 p-4">
                      <div className="grid gap-4 md:grid-cols-2">
                        <div>
                          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            Before
                            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                              HTML
                            </span>
                          </div>
                          <p className="mt-2 text-xs font-mono text-muted-foreground">
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;P STYLE="margin-top:12pt; margin-bottom:0pt;
                              text-indent:4%; font-size:10pt; font-family:Times
                              New Roman"&gt;
                            </span>
                            Section&nbsp;4.14{" "}
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;U&gt;
                            </span>
                            Brokers and Other Advisors
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;/U&gt;
                            </span>
                            . Except for Citigroup Global Markets Inc., the fees
                            and expenses of which will be paid by ETE, no
                            broker, investment banker or financial advisor is
                            entitled to any broker&#146;s, finder&#146;s or
                            financial advisor&#146;s fee or commission, or the
                            reimbursement of expenses, in connection with the
                            Merger or the transactions contemplated hereby based
                            on arrangements made by or on behalf of ETE GP, ETE
                            or any of their respective Subsidiaries.{" "}
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;/P&gt;
                            </span>
                          </p>
                        </div>
                        <div>
                          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            After
                            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                              Plain Text
                            </span>
                          </div>
                          <p className="mt-2 text-sm text-foreground">
                            Section 4.14 Brokers and Other Advisors. Except for
                            Citigroup Global Markets Inc., the fees and expenses
                            of which will be paid by ETE, no broker, investment
                            banker or financial advisor is entitled to any
                            broker’s, finder’s or financial advisor’s fee or
                            commission, or the reimbursement of expenses, in
                            connection with the Merger or the transactions
                            contemplated hereby based on arrangements made by or
                            on behalf of ETE GP, ETE or any of their respective
                            Subsidiaries.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
                <li className="relative flex gap-4">
                  <div className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70">
                    3
                  </div>
                  <div className="space-y-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-foreground">
                        Tagging
                      </span>
                      <a
                        href="#page-tagging"
                        className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-muted-foreground transition-colors hover:border-emerald-500/40 hover:text-foreground"
                      >
                        Tagging Model
                      </a>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Applies the tagging model to{" "}
                      <span className="font-mono text-xs text-foreground">
                        body
                      </span>{" "}
                      pages, marking articles, sections, and page numbers.
                    </p>
                    <div className="rounded-xl border border-border/60 bg-background/60 p-4">
                      <div className="grid gap-4 md:grid-cols-2">
                        <div>
                          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            Before
                            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                              Plain Text
                            </span>
                          </div>
                          <p className="mt-2 text-sm text-muted-foreground">
                            Section 4.14 Brokers and Other Advisors. Except for
                            Citigroup Global Markets Inc., the fees and expenses
                            of which will be paid by ETE, no broker, investment
                            banker or financial advisor is entitled to any
                            broker’s, finder’s or financial advisor’s fee or
                            commission, or the reimbursement of expenses, in
                            connection with the Merger or the transactions
                            contemplated hereby based on arrangements made by or
                            on behalf of ETE GP, ETE or any of their respective
                            Subsidiaries.
                          </p>
                        </div>
                        <div>
                          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            After
                            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                              XML
                            </span>
                          </div>
                          <p className="mt-2 text-xs font-mono text-foreground">
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;section&gt;Section 4.14 Brokers and Other
                              Advisors.&lt;/section&gt;
                            </span>{" "}
                            Except for Citigroup Global Markets Inc., the fees
                            and expenses of which will be paid by ETE, no
                            broker, investment banker or financial advisor is
                            entitled to any broker’s, finder’s or financial
                            advisor’s fee or commission, or the reimbursement of
                            expenses, in connection with the Merger or the
                            transactions contemplated hereby based on
                            arrangements made by or on behalf of ETE GP, ETE or
                            any of their respective Subsidiaries.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
                <li className="relative flex gap-4">
                  <div className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70">
                    4
                  </div>
                  <div className="space-y-2">
                    <div className="font-mono text-sm font-semibold text-foreground">
                      LLM Review
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Sends low-confidence spans to{" "}
                      <span className="font-mono text-xs text-foreground">
                        gpt-5
                      </span>{" "}
                      with adjacent context so the model can rule on the correct
                      tag.
                    </p>
                  </div>
                </li>
                <li className="relative flex gap-4">
                  <div className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70">
                    5
                  </div>
                  <div className="space-y-2">
                    <div className="font-mono text-sm font-semibold text-foreground">
                      Reconciliation
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Merges LLM-corrected spans back into the tagged agreement
                      text.
                    </p>
                  </div>
                </li>
                <li className="relative flex gap-4">
                  <div className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70">
                    6
                  </div>
                  <div className="space-y-2">
                    <div className="font-mono text-sm font-semibold text-foreground">
                      XML Construction
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Builds{" "}
                      <span className="font-mono text-xs text-foreground">
                        xml
                      </span>{" "}
                      output and upserts section records into the sections
                      table.
                    </p>
                  </div>
                </li>
                <li className="relative flex gap-4">
                  <div className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70">
                    7
                  </div>
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-foreground">
                        Taxonomy
                      </span>
                      <a
                        href="#section-classification"
                        className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-muted-foreground transition-colors hover:border-emerald-500/40 hover:text-foreground"
                      >
                        Taxonomy Model
                      </a>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Classifies each section into the clause taxonomy for
                      cross-agreement analysis.
                    </p>
                  </div>
                </li>
                <li className="relative flex gap-4">
                  <div
                    ref={lastStepRef}
                    className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70"
                  >
                    8
                  </div>
                  <div className="space-y-2">
                    <div className="font-mono text-sm font-semibold text-foreground">
                      Metadata
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Adds transaction metadata, including total consideration,
                      consideration types, and target and acquirer industries.
                    </p>
                  </div>
                </li>
              </ol>
            </div>
          </section>

          <section id="ml-models" className="scroll-mt-24 space-y-4">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              ML Models
            </h2>
            <div className="text-muted-foreground">
              While we'd prefer to outsource all inference tasks to an LLM,
              Pandects is a small-scale operation run on a shoe-string budget,
              meaning we don't have the resources to send hundreds of thousands
              of pages of agreements to proprietary LLMs—and as of October 2025,{" "}
              {isCoarsePointer ? (
                <Popover>
                  <PopoverTrigger asChild>
                    <button
                      type="button"
                      className="cursor-help appearance-none bg-transparent p-0 text-inherit underline decoration-dotted underline-offset-4"
                    >
                      open-source models
                    </button>
                  </PopoverTrigger>
                  <PopoverContent
                    side="top"
                    className="max-w-xs border-border/70 bg-background/95 text-xs text-foreground shadow-lg"
                  >
                    We tried DeepSeek R1 on NYU's HPC clusters, and the results
                    were less than reliable.
                  </PopoverContent>
                </Popover>
              ) : (
                <Tooltip delayDuration={300}>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      className="cursor-help appearance-none bg-transparent p-0 text-inherit underline decoration-dotted underline-offset-4"
                    >
                      open-source models
                    </button>
                  </TooltipTrigger>
                  <TooltipContent
                    side="top"
                    className="max-w-xs border-border/70 bg-background/95 text-xs text-foreground shadow-lg"
                  >
                    We tried DeepSeek R1 on NYU's HPC clusters, and the results
                    were less than reliable.
                  </TooltipContent>
                </Tooltip>
              )}{" "}
              still made too many mistakes for us to be comfortable using them
              at scale. Our solution: use the latest and greatest LLMs to
              generate high-quality training data from a limited sample of
              agreements—<strong>137</strong> in total, sampled roughly evenly
              from 2000 to 2020—and then train up more routine machine learning
              models to do the work we'd otherwise outsource to LLMs.
            </div>
            <p className="text-muted-foreground">
              This section describes the four ML models at the heart of our data
              processing pipeline.
            </p>
            <div
              id="agreement-identification"
              className="scroll-mt-24 pt-2 space-y-4"
            >
              <h3 className="text-lg font-semibold text-foreground">
                Agreement Model
              </h3>
              <ComingSoon title="Matching and deduplication logic" />
            </div>
            <div
              id="page-classification"
              className="scroll-mt-24 pt-2 space-y-4"
            >
              <h3 className="text-lg font-semibold text-foreground">
                Page Classifier Model
              </h3>
              <div className="text-muted-foreground">
                The Page Classifier Model combines two layers: 1) an XGBoost
                model that scores each page <em>individually</em>, using
                engineered text/HTML/layout features; and 2) a BiLSTM +
                constrained CRF that refines those scores into a coherent,
                document-level page sequence. The CRF encodes ordering rules
                (including a single contiguous signature block), while learned
                positional priors and auxiliary onset losses help place key
                transitions reliably. We trained both models on 80% random
                splits of the full set of{" "}
                {isCoarsePointer ? (
                  <Popover>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        className="cursor-help appearance-none bg-transparent p-0 text-inherit underline decoration-dotted underline-offset-4"
                      >
                        <strong>11,854</strong>
                      </button>
                    </PopoverTrigger>
                    <PopoverContent
                      side="top"
                      className="max-w-xs border-border/70 bg-background/95 text-xs text-foreground shadow-lg"
                    >
                      Did we really label 11,854 pages by hand? Yes—but it wasn't
                      that bad. We had GPT build us a custom labeling interface
                      that allowed us to select all body pages in a single go, so
                      the whole process took less than three hours.
                    </PopoverContent>
                  </Popover>
                ) : (
                  <Tooltip delayDuration={300}>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        className="cursor-help appearance-none bg-transparent p-0 text-inherit underline decoration-dotted underline-offset-4"
                      >
                        <strong>11,854</strong>
                      </button>
                    </TooltipTrigger>
                    <TooltipContent
                      side="top"
                      className="max-w-xs border-border/70 bg-background/95 text-xs text-foreground shadow-lg"
                    >
                      Did we really label 11,854 pages by hand? Yes—but it wasn't
                      that bad. We had GPT build us a custom labeling interface
                      that allowed us to select all body pages in a single go, so
                      the whole process took less than three hours.
                    </TooltipContent>
                  </Tooltip>
                )}{" "}
                manually labeled pages. The full training code is available on{" "}
                <a
                  href="https://github.com/PlatosTwin/pandects-app/blob/main/etl/src/etl/models/code/classifier.py"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                GitHub
              </a>
              .
              </div>
              <p className="text-muted-foreground">
                Below are model performance metrics for both the baseline XGB
                model and the full BiLSTM model. While the XGB model does
                reasonably well on its own, taking into account and enforcing
                page order substantially improves performance, bringing the F1
                score up from <strong>94.36%</strong> to <strong>99.69%</strong>
                .{" "}
                <em>
                  Note that both sets of metrics are based on validation
                  splits—16% for the baseline model and 20% for the final
                  model—rather than a true test split.
                </em>
              </p>
              <div className="space-y-6">
                <div className="rounded-2xl border border-border/70 bg-card/60 p-5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-foreground">
                      Model Metrics
                    </div>
                    <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                      XGB Baseline
                    </span>
                  </div>
                  <div className="mt-4 grid gap-6 lg:grid-cols-2">
                    <div className="rounded-lg bg-emerald-500/10 p-3 lg:col-span-2">
                      <div className="grid gap-3 text-emerald-900 dark:text-emerald-100 sm:grid-cols-2 lg:grid-cols-4">
                        <div className="text-center sm:text-left">
                          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
                            Accuracy
                          </div>
                          <div className="mt-1 text-2xl font-semibold text-foreground">
                            {formatMetric(0.9657)}
                          </div>
                        </div>
                        <div className="text-center sm:text-left">
                          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
                            Precision
                          </div>
                          <div className="mt-1 text-2xl font-semibold text-foreground">
                            {formatMetric(0.9272)}
                          </div>
                        </div>
                        <div className="text-center sm:text-left">
                          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
                            Recall
                          </div>
                          <div className="mt-1 text-2xl font-semibold text-foreground">
                            {formatMetric(0.9621)}
                          </div>
                        </div>
                        <div className="text-center sm:text-left">
                          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
                            F1 Score
                          </div>
                          <div className="mt-1 text-2xl font-semibold text-foreground">
                            {formatMetric(0.9436)}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
                      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        Confusion Matrix
                      </div>
                      <div className="mt-3 space-y-2">
                        <div className="text-center text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                          Predicted
                        </div>
                        <div className="w-full max-w-full overflow-x-auto">
                          <div className="flex items-stretch gap-0">
                            <div className="relative w-0">
                              <span className="absolute right-1 top-1/2 -translate-y-1/2 -rotate-90 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                                Actual
                              </span>
                            </div>
                            <table className="w-full min-w-[320px] table-fixed border-separate border-spacing-1 text-[11px]">
                              <colgroup>
                                <col className="w-8" />
                                {classifierAbbreviations.map((label) => (
                                  <col key={label} className="w-12" />
                                ))}
                              </colgroup>
                              <caption className="sr-only">
                                XGB baseline confusion matrix
                              </caption>
                              <thead>
                                <tr>
                                  <th
                                    aria-hidden="true"
                                    className="p-1 text-left text-muted-foreground"
                                  />
                                  {classifierAbbreviations.map((label) => (
                                    <th
                                      key={label}
                                      scope="col"
                                      className="p-1 text-center font-mono text-muted-foreground"
                                    >
                                      {label}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {pageClassifierMatrix.map((row, rowIndex) => (
                                  <tr key={classifierLabels[rowIndex]}>
                                    <th
                                      scope="row"
                                      className="p-1 pl-0 text-left font-mono text-muted-foreground"
                                    >
                                      {classifierAbbreviations[rowIndex]}
                                    </th>
                                    {row.map((value, colIndex) => {
                                      const isDiagonal = rowIndex === colIndex;
                                      const hasValue = value > 0;
                                      const cellClass = isDiagonal
                                        ? "bg-emerald-500/20 text-foreground"
                                        : hasValue
                                          ? "bg-rose-500/15 text-foreground"
                                          : "bg-muted/40 text-muted-foreground/60";
                                      return (
                                        <td
                                          key={`${rowIndex}-${colIndex}`}
                                          className={`rounded-md px-2 py-1 text-center font-mono ${cellClass}`}
                                        >
                                          {value}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
                      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        Per-class Metrics
                      </div>
                      <div className="mt-3 w-full overflow-x-auto">
                        <table className="w-full min-w-[320px] text-xs">
                          <caption className="sr-only">
                            XGB baseline per-class metrics
                          </caption>
                          <thead>
                            <tr className="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
                              <th scope="col" className="pb-2 pr-3">
                                Class
                              </th>
                              <th scope="col" className="pb-2 pr-3 text-right">
                                Acc
                              </th>
                              <th scope="col" className="pb-2 pr-3 text-right">
                                P
                              </th>
                              <th scope="col" className="pb-2 pr-3 text-right">
                                R
                              </th>
                              <th scope="col" className="pb-2 text-right">
                                F1
                              </th>
                            </tr>
                          </thead>
                          <tbody className="font-mono text-muted-foreground">
                            {pageClassifierMetrics.map((metric) => {
                              const statusClass =
                                metric.f1 >= 0.95
                                  ? "bg-emerald-500"
                                  : metric.f1 < 0.9
                                    ? "bg-amber-400"
                                    : "bg-muted-foreground/40";
                              return (
                                <tr
                                  key={metric.label}
                                  className="border-b border-border/40"
                                >
                                  <th
                                    scope="row"
                                    className="py-2 pr-3 text-left text-foreground font-normal"
                                  >
                                    <span
                                      className={`mr-2 inline-flex h-2 w-2 rounded-full ${statusClass}`}
                                    />
                                    {classifierAbbreviations[
                                      classifierLabels.indexOf(metric.label)
                                    ] ?? metric.label}
                                  </th>
                                  <td className="py-2 pr-3 text-right">
                                    {formatMetric(metric.acc)}
                                  </td>
                                  <td className="py-2 pr-3 text-right">
                                    {formatMetric(metric.p)}
                                  </td>
                                  <td className="py-2 pr-3 text-right">
                                    {formatMetric(metric.r)}
                                  </td>
                                  <td className="py-2 text-right">
                                    {formatMetric(metric.f1)}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="rounded-2xl border border-border/70 bg-card/60 p-5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-foreground">
                      Final Classifier Metrics
                    </div>
                    <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                      CRF Final
                    </span>
                  </div>
                  <div className="mt-4 grid gap-6 lg:grid-cols-2">
                    <div className="rounded-lg bg-emerald-500/10 p-3 lg:col-span-2">
                      <div className="grid gap-3 text-emerald-900 dark:text-emerald-100 sm:grid-cols-2 lg:grid-cols-4">
                        <div className="text-center sm:text-left">
                          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
                            Accuracy
                          </div>
                          <div className="mt-1 text-2xl font-semibold text-foreground">
                            {formatMetric(0.9996)}
                          </div>
                        </div>
                        <div className="text-center sm:text-left">
                          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
                            Precision
                          </div>
                          <div className="mt-1 text-2xl font-semibold text-foreground">
                            {formatMetric(0.999)}
                          </div>
                        </div>
                        <div className="text-center sm:text-left">
                          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
                            Recall
                          </div>
                          <div className="mt-1 text-2xl font-semibold text-foreground">
                            {formatMetric(0.9949)}
                          </div>
                        </div>
                        <div className="text-center sm:text-left">
                          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
                            F1 Score
                          </div>
                          <div className="mt-1 text-2xl font-semibold text-foreground">
                            {formatMetric(0.9969)}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
                      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        Confusion Matrix
                      </div>
                      <div className="mt-3 space-y-2">
                        <div className="text-center text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                          Predicted
                        </div>
                        <div className="w-full max-w-full overflow-x-auto">
                          <div className="flex items-stretch gap-0">
                            <div className="relative w-0">
                              <span className="absolute right-1 top-1/2 -translate-y-1/2 -rotate-90 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                                Actual
                              </span>
                            </div>
                            <table className="w-full min-w-[320px] table-fixed border-separate border-spacing-1 text-[11px]">
                              <colgroup>
                                <col className="w-8" />
                                {finalClassifierAbbreviations.map((label) => (
                                  <col key={label} className="w-12" />
                                ))}
                              </colgroup>
                              <caption className="sr-only">
                                Final classifier confusion matrix
                              </caption>
                              <thead>
                                <tr>
                                  <th
                                    aria-hidden="true"
                                    className="p-1 text-left text-muted-foreground"
                                  />
                                  {finalClassifierAbbreviations.map((label) => (
                                    <th
                                      key={label}
                                      scope="col"
                                      className="p-1 text-center font-mono text-muted-foreground"
                                    >
                                      {label}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {finalClassifierMatrix.map((row, rowIndex) => (
                                  <tr key={finalClassifierLabels[rowIndex]}>
                                    <th
                                      scope="row"
                                      className="p-1 pl-0 text-left font-mono text-muted-foreground"
                                    >
                                      {finalClassifierAbbreviations[rowIndex]}
                                    </th>
                                    {row.map((value, colIndex) => {
                                      const isDiagonal = rowIndex === colIndex;
                                      const hasValue = value > 0;
                                      const cellClass = isDiagonal
                                        ? "bg-emerald-500/20 text-foreground"
                                        : hasValue
                                          ? "bg-rose-500/15 text-foreground"
                                          : "bg-muted/40 text-muted-foreground/60";
                                      return (
                                        <td
                                          key={`${rowIndex}-${colIndex}`}
                                          className={`rounded-md px-2 py-1 text-center font-mono ${cellClass}`}
                                        >
                                          {value}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
                      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        Per-class Metrics
                      </div>
                      <div className="mt-3 w-full overflow-x-auto">
                        <table className="w-full min-w-[320px] text-xs">
                          <caption className="sr-only">
                            Final classifier per-class metrics
                          </caption>
                          <thead>
                            <tr className="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
                              <th scope="col" className="pb-2 pr-3">
                                Class
                              </th>
                              <th scope="col" className="pb-2 pr-3 text-right">
                                Acc
                              </th>
                              <th scope="col" className="pb-2 pr-3 text-right">
                                P
                              </th>
                              <th scope="col" className="pb-2 pr-3 text-right">
                                R
                              </th>
                              <th scope="col" className="pb-2 text-right">
                                F1
                              </th>
                            </tr>
                          </thead>
                          <tbody className="font-mono text-muted-foreground">
                            {finalClassifierMetrics.map((metric) => {
                              const statusClass =
                                metric.f1 >= 0.95
                                  ? "bg-emerald-500"
                                  : metric.f1 < 0.9
                                    ? "bg-amber-400"
                                    : "bg-muted-foreground/40";
                              return (
                                <tr
                                  key={metric.label}
                                  className="border-b border-border/40"
                                >
                                  <th
                                    scope="row"
                                    className="py-2 pr-3 text-left text-foreground font-normal"
                                  >
                                    <span
                                      className={`mr-2 inline-flex h-2 w-2 rounded-full ${statusClass}`}
                                    />
                                    {finalClassifierAbbreviations[
                                      finalClassifierLabels.indexOf(
                                        metric.label
                                      )
                                    ] ?? metric.label}
                                  </th>
                                  <td className="py-2 pr-3 text-right">
                                    {formatMetric(metric.acc)}
                                  </td>
                                  <td className="py-2 pr-3 text-right">
                                    {formatMetric(metric.p)}
                                  </td>
                                  <td className="py-2 pr-3 text-right">
                                    {formatMetric(metric.r)}
                                  </td>
                                  <td className="py-2 text-right">
                                    {formatMetric(metric.f1)}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div id="page-tagging" className="scroll-mt-24 pt-2 space-y-4">
              <h3 className="text-lg font-semibold text-foreground">
                Tagging Model
              </h3>
              <p>
                The Tagging Model fine-tunes{" "}
                <a
                  href="https://huggingface.co/blog/modernbert"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  <span className="font-mono text-sm text-foreground">
                    ModernBERT-base
                  </span>
                </a>{" "}
                for token-level labeling using a BIOES tagging scheme, producing
                structured tags for sections, articles, and pages. It trains
                with focal loss and warmup scheduling, and evaluates
                entity-level accuracy by stitching overlapping token windows and
                averaging logits across them. At inference time, a constrained
                Viterbi decoder enforces legal BIOES transitions, yielding
                cleaner, more consistent spans across long documents. The full
                training code is available on{" "}
                <a
                  href="https://github.com/PlatosTwin/pandects-app/blob/main/etl/src/etl/models/code/ner.py"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  GitHub
                </a>
                .
              </p>
            </div>
            <div
              id="section-classification"
              className="scroll-mt-24 pt-2 space-y-4"
            >
              <h3 className="text-lg font-semibold text-foreground">
                Taxonomy Model
              </h3>
              <ComingSoon title="Mapping to the Pandects taxonomy" />
            </div>
          </section>

          <section id="gaps-and-callouts" className="scroll-mt-24 space-y-4">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Gaps and Other Call Outs
            </h2>
            <ComingSoon title="Known limitations and edge cases" />
          </section>

          <section id="validations" className="scroll-mt-24 space-y-4">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Validations
            </h2>
            <ComingSoon title="Accuracy checks and evaluation results" />
          </section>
        </div>
      </div>
    </PageShell>
  );
}
