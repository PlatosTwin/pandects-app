import type { ReactNode } from "react";
import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import ResizeObserverPolyfill from "resize-observer-polyfill";
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
import type {
  NerEvalData,
  ClassifierEvalData,
  ExhibitEvalData,
} from "@/lib/model-metrics-types";

const LazyClassifierEvalMetrics = lazy(() =>
  import("@/components/ClassifierEvalMetrics").then((mod) => ({
    default: mod.ClassifierEvalMetrics,
  }))
);
const LazyNerEvalMetrics = lazy(() =>
  import("@/components/NerEvalMetrics").then((mod) => ({
    default: mod.NerEvalMetrics,
  }))
);
const LazyExhibitEvalMetrics = lazy(() =>
  import("@/components/ExhibitEvalMetrics").then((mod) => ({
    default: mod.ExhibitEvalMetrics,
  }))
);

type MetricsData = {
  classifier: ClassifierEvalData;
  ner: NerEvalData;
  exhibit: ExhibitEvalData;
};

type InfoDisclosureProps = {
  isCoarsePointer: boolean;
  label: ReactNode;
  ariaLabel?: string;
  triggerClassName?: string;
  content: ReactNode;
};

function InfoDisclosure({
  isCoarsePointer,
  label,
  ariaLabel,
  triggerClassName,
  content,
}: InfoDisclosureProps) {
  if (isCoarsePointer) {
    return (
      <Popover>
        <PopoverTrigger asChild>
          <button
            type="button"
            aria-label={ariaLabel}
            className={triggerClassName}
          >
            {label}
          </button>
        </PopoverTrigger>
        <PopoverContent
          side="top"
          className="max-w-xs border-border/70 bg-background/95 text-xs text-foreground shadow-lg"
        >
          {content}
        </PopoverContent>
      </Popover>
    );
  }

  return (
    <Tooltip delayDuration={300}>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={ariaLabel}
          className={triggerClassName}
        >
          {label}
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        className="max-w-xs border-border/70 bg-background/95 text-xs text-foreground shadow-lg"
      >
        {content}
      </TooltipContent>
    </Tooltip>
  );
}

export default function SourcesMethods() {
  const [activeSection, setActiveSection] = useState("");
  const firstStepRef = useRef<HTMLDivElement | null>(null);
  const lastStepRef = useRef<HTMLDivElement | null>(null);
  const pipelineRef = useRef<HTMLDivElement | null>(null);
  const progressRef = useRef<HTMLDivElement | null>(null);
  const [pipelineLine, setPipelineLine] = useState({ top: 0, height: 0 });
  const [isCoarsePointer, setIsCoarsePointer] = useState(false);
  const pipelineMetricsRef = useRef<{
    firstCenter: number;
    lastCenter: number;
    lineTop: number;
    lineHeight: number;
  } | null>(null);
  const metricsRef = useRef<HTMLDivElement | null>(null);
  const [shouldLoadMetrics, setShouldLoadMetrics] = useState(false);
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);

  const navItems = useMemo(
    () => [
      { id: "overview", label: "Overview" },
      { id: "data-pipeline-architecture", label: "Data Pipeline Architecture" },
      { id: "ml-models", label: "ML Models" },
      {
        id: "exhibit-model",
        label: "Exhibit Model",
        indent: true,
      },
      {
        id: "page-classifier-model",
        label: "Page Classifier Model",
        indent: true,
      },
      { id: "tagging-model", label: "Tagging Model", indent: true },
      {
        id: "taxonomy-model",
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

    // Calculate target position manually to avoid scrollIntoView inconsistencies
    // with scroll-margin-top during concurrent re-renders
    const rect = el.getBoundingClientRect();
    const scrollMarginTop =
      parseFloat(getComputedStyle(el).scrollMarginTop) || 0;
    const targetY = window.scrollY + rect.top - scrollMarginTop;

    window.scrollTo({
      top: targetY,
      behavior: prefersReducedMotion ? "auto" : "smooth",
    });

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
      const metrics = pipelineMetricsRef.current;
      if (!metrics) return;
      const denom = metrics.lastCenter - metrics.firstCenter;
      if (!denom) return;
      const viewportHeight = window.innerHeight || 1;
      const viewportMid = window.scrollY + viewportHeight / 2;
      const rawProgress = (viewportMid - metrics.firstCenter) / denom;
      const clamped = Math.min(1, Math.max(0, rawProgress));
      if (progressRef.current) {
        progressRef.current.style.transform = `scaleY(${clamped})`;
      }
    };

    let raf = 0;
    const onScroll = () => {
      if (raf) return;
      raf = window.requestAnimationFrame(() => {
        raf = 0;
        updateProgress();
      });
    };

    const updateMetrics = () => {
      const firstEl = firstStepRef.current;
      const lastEl = lastStepRef.current;
      if (!firstEl || !lastEl) return;
      const firstRect = firstEl.getBoundingClientRect();
      const lastRect = lastEl.getBoundingClientRect();
      if (!firstRect.height || !lastRect.height) return;
      const containerRect = pipelineRef.current?.getBoundingClientRect();
      const scrollY = window.scrollY;
      const firstCenter = firstRect.top + scrollY + firstRect.height / 2;
      const lastCenter = lastRect.top + scrollY + lastRect.height / 2;
      let lineTop = 0;
      let lineHeight = 0;
      if (containerRect) {
        lineTop = firstRect.top - containerRect.top + firstRect.height / 2;
        const bottom = lastRect.top - containerRect.top + lastRect.height / 2;
        lineHeight = Math.max(0, bottom - lineTop);
        setPipelineLine((prev) =>
          Math.abs(prev.top - lineTop) > 0.5 ||
          Math.abs(prev.height - lineHeight) > 0.5
            ? { top: lineTop, height: lineHeight }
            : prev
        );
      }
      pipelineMetricsRef.current = {
        firstCenter,
        lastCenter,
        lineTop,
        lineHeight,
      };
      // Don't call updateProgress directly inside updateMetrics to avoid potential layout thrashing
      // Instead, just ensure the next scroll/frame picks up the new metrics
    };

    let metricsRaf = 0;
    const scheduleMetrics = () => {
      if (metricsRaf) return;
      metricsRaf = window.requestAnimationFrame(() => {
        metricsRaf = 0;
        updateMetrics();
      });
    };

    updateMetrics();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", scheduleMetrics);
    const Observer =
      typeof ResizeObserver !== "undefined"
        ? ResizeObserver
        : ResizeObserverPolyfill;
    const observer = new Observer(scheduleMetrics);
    if (observer) {
      if (pipelineRef.current) observer.observe(pipelineRef.current);
      if (firstStepRef.current) observer.observe(firstStepRef.current);
      if (lastStepRef.current) observer.observe(lastStepRef.current);
    }
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", scheduleMetrics);
      if (observer) observer.disconnect();
      if (raf) window.cancelAnimationFrame(raf);
      if (metricsRaf) window.cancelAnimationFrame(metricsRaf);
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

  useEffect(() => {
    if (shouldLoadMetrics) return;
    if (typeof window === "undefined") return;
    const target = metricsRef.current;
    if (!target) return;

    if (!("IntersectionObserver" in window)) {
      setShouldLoadMetrics(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setShouldLoadMetrics(true);
          observer.disconnect();
        }
      },
      { rootMargin: "400px" }
    );

    observer.observe(target);
    return () => observer.disconnect();
  }, [shouldLoadMetrics]);

  useEffect(() => {
    if (!shouldLoadMetrics) return;
    let cancelled = false;

    const load = () => {
      void import("./sources-methods-metrics").then((mod) => {
        if (cancelled) return;
        setMetricsData({
          classifier: mod.classifierEvalData,
          ner: mod.nerEvalData,
          exhibit: mod.exhibitEvalData,
        });
      });
    };

    const schedule = window.requestIdleCallback
      ? window.requestIdleCallback(load, { timeout: 1800 })
      : window.setTimeout(load, 800);

    return () => {
      cancelled = true;
      if (window.cancelIdleCallback) {
        window.cancelIdleCallback(schedule as number);
      } else {
        window.clearTimeout(schedule as number);
      }
    };
  }, [shouldLoadMetrics]);

  useEffect(() => {
    if (!metricsRef.current) return;

    const handleResize = () => {
      if (
        [
          "exhibit-model",
          "page-classifier-model",
          "tagging-model",
          "taxonomy-model",
        ].includes(activeSection)
      ) {
        scrollToSection(activeSection);
      }
    };

    const Observer =
      typeof ResizeObserver !== "undefined"
        ? ResizeObserver
        : ResizeObserverPolyfill;

    const observer = new Observer(handleResize);
    observer.observe(metricsRef.current);

    return () => observer.disconnect();
  }, [activeSection]);

  const ComingSoon = ({ title }: { title: string }) => (
    <Card className="border-border/70 bg-card/70 p-5">
      <div className="text-sm font-medium text-foreground">{title}</div>
      <p className="mt-1 text-sm text-muted-foreground">
        This section is being written. If you would like to help, open an issue
        with suggestions.
      </p>
    </Card>
  );

  const baselineClassifierF1 = 0.9270384433262334;
  const postProcessingF1 = 0.9553224497648898;
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

        <div className="space-y-12 min-w-0">
          <section id="overview" className="scroll-mt-32 space-y-4">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Overview
            </h2>
            <p className="text-muted-foreground">
              Pandects sources agreements from the SEC's EDGAR database, then
              runs each agreement through a purpose-built pipeline that turns
              text and messy HTML into clean, taxonomized XML. Conceptually, our
              pipeline solves five distinct problems.
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
                        We parse through Exhibit 2 and Exhibit 10 filings at
                        scale and use machine learning to identify those that
                        represent definitive merger agreements.
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
                        We use machine learning to normalize every agreement
                        into standardized XML, stripping away the HTML mess so
                        researchers don't have to invent their own parsing
                        logic.
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
                        challenging to classify clauses at scale, and existing
                        datasets don't offer taxonomies.
                      </p>
                    </div>
                    <div className="grid content-start gap-2 rounded-lg bg-emerald-500/10 px-2 py-3 md:col-start-3 md:row-span-2">
                      <span className="sr-only">Pandects standard:</span>
                      <div className="text-sm font-semibold text-foreground">
                        Rigorous Taxonomy
                      </div>
                      <p className="text-sm text-foreground">
                        We use machine learning to classify every section into a
                        specific clause taxonomy, making it effortless to query
                        specific clause types across thousands of deals.
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
                        deal value, industry codes, and buyer types, and
                        existing datasets don't provide this metadata.
                      </p>
                    </div>
                    <div className="grid content-start gap-2 rounded-lg bg-emerald-500/10 px-2 py-3 md:col-start-3 md:row-span-2">
                      <span className="sr-only">Pandects standard:</span>
                      <div className="text-sm font-semibold text-foreground">
                        LLM-Enriched Data
                      </div>
                      <p className="text-sm text-foreground">
                        We use LLMs to extract and incorporate deal-specific
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
            className="scroll-mt-32 space-y-4"
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
                className="text-primary underline underline-offset-2 hover:underline"
              >
                Dagster
              </a>{" "}
              to orchestrate the pipeline, which is defined in full on{" "}
              <a
                href="https://github.com/PlatosTwin/pandects-app/tree/main/etl/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline underline-offset-2 hover:underline"
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
                  ref={progressRef}
                  className="absolute left-0 top-0 h-full w-full origin-top bg-gradient-to-b from-emerald-300 via-emerald-500 to-emerald-600 shadow-[0_0_10px_rgba(16,185,129,0.35)] will-change-transform"
                  style={{ transform: "scaleY(0)", willChange: "transform" }}
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
                        href="#exhibit-model"
                        className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-foreground transition-colors hover:border-emerald-500/40"
                      >
                        Exhibit Model
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
                      <InfoDisclosure
                        isCoarsePointer={isCoarsePointer}
                        ariaLabel="Why we split and classify pages"
                        triggerClassName="inline-flex h-5 w-5 items-center justify-center rounded-full border border-border/70 bg-muted/40 text-[10px] font-semibold text-muted-foreground transition-colors hover:border-emerald-500/40 hover:text-foreground cursor-help"
                        label="?"
                        content={
                          <>
                            Why split agreements into pages? Because our NER
                            model has a limited context window, meaning we'd
                            have to chunk agreement text regardless, and page
                            markers are built-in split points. Why categorize
                            pages into classes? Primarily to identify body
                            pages, which form the core of agreements, and to
                            increase the accuracy of our Tagging Model, which
                            likely would struggle with the structural variety of
                            appendices and exhibit sections.
                          </>
                        }
                      />
                      <a
                        href="#page-classifier-model"
                        className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-foreground transition-colors hover:border-emerald-500/40"
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
                        href="#tagging-model"
                        className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-foreground transition-colors hover:border-emerald-500/40"
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
                        href="#taxonomy-model"
                        className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-foreground transition-colors hover:border-emerald-500/40"
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

          <section
            id="ml-models"
            ref={metricsRef}
            className="scroll-mt-32 space-y-4"
          >
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              ML Models
            </h2>
            <div className="text-muted-foreground">
              While we'd prefer to outsource all inference tasks to an LLM,
              Pandects is a small-scale operation run on a shoestring budget,
              meaning we don't have the resources to send hundreds of thousands
              of pages of agreements to proprietary LLMs—and as of October 2025,{" "}
              <InfoDisclosure
                isCoarsePointer={isCoarsePointer}
                triggerClassName="cursor-help appearance-none bg-transparent p-0 text-inherit underline decoration-dotted underline-offset-4"
                label="open-source models"
                content={
                  <>
                    We tried DeepSeek R1 on NYU's HPC clusters, and the results
                    were less than reliable.
                  </>
                }
              />{" "}
              still made too many mistakes for us to be comfortable using them
              at scale. Our solution: use the latest and greatest LLMs to
              generate high-quality training data from a limited sample of
              agreements and then train up more routine machine learning models
              to do the work we'd otherwise outsource to LLMs.
            </div>
            <p className="text-muted-foreground">
              This section describes the four ML models at the heart of our data
              processing pipeline. The full training code for each model is
              available on{" "}
              <a
                href="https://github.com/PlatosTwin/pandects-app/blob/main/etl/src/etl/models/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline underline-offset-2 hover:underline"
              >
                GitHub
              </a>
              .
            </p>
            <div
              id="exhibit-model"
              className="scroll-mt-32 pt-2 space-y-4"
            >
              <h3 className="text-lg font-semibold text-foreground">
                Exhibit Model
              </h3>
              <div className="text-muted-foreground">
                The Exhibit Model takes as inputs filing from the SEC's EDGAR
                database, and for each filing outputs the probability that that
                filing represents a definitive merger agreement. To identify
                filings, we scan the{" "}
                <a
                  href="https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data#:~:text=Using%20the%20EDGAR%20index%20files"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline underline-offset-2 hover:underline"
                >
                  SEC's daily index files
                </a>
                , identify all filings that match target form types (all forms
                except for S-8 and ABS-EE), filter filings by scanning for
                keywords indicating "Material Definitive Agreement" entries,
                scrape each filing's index page to extract Exhibit 10.* and 2.*
                links, fetch and render the exhibit content as text, and use the
                Exhibit Model to classify candidates, retaining only those with
                probability of at least 0.5.
              </div>
              <ul className="list-disc ml-6 space-y-4 text-muted-foreground">
                <li className="space-y-2">
                  <div>
                    <strong>Architecture:</strong> The Exhibit Model is a simple
                    binary classifier that uses logistic regression. As
                    features, we use: (1) <em>document-level features</em>{" "}
                    including structural indicators, legal language patterns,
                    and M&A-specific vocabulary; (2) <em>TF-IDF features</em>{" "}
                    from word and character n-grams; and (3){" "}
                    <em>similarity features</em> measuring cosine similarity to
                    training examples.
                  </div>
                </li>
                <li className="space-y-2">
                  <div>
                    <strong>Training corpus:</strong> We train the Exhibit Model
                    on an 80/20 random split of <strong>1,392</strong> positive
                    samples, selected from the DMA corpus, and 250 negatives,
                    selected from SEC filings under Exhibits 2 and 10, with a
                    bias toward filings that earlier versions of this model
                    incorrectly identified as agreements.
                  </div>
                </li>
              </ul>
              {metricsData ? (
                <Suspense
                  fallback={
                    <Card className="border-border/70 bg-card/70 p-5 text-sm text-muted-foreground">
                      Loading model metrics...
                    </Card>
                  }
                >
                  <LazyExhibitEvalMetrics data={metricsData.exhibit} />
                </Suspense>
              ) : (
                <Card className="border-border/70 bg-card/70 p-5 text-sm text-muted-foreground">
                  Loading model metrics...
                </Card>
              )}
            </div>
            <div
              id="page-classifier-model"
              className="scroll-mt-32 pt-2 space-y-4"
            >
              <h3 className="text-lg font-semibold text-foreground">
                Page Classifier Model
              </h3>
              <div className="text-muted-foreground">
                The Page Classifier takes as inputs pages from filings
                identified as definitive merger agreements and outputs for each
                page the probability that that page belongs to each one of five
                classes, along with the actual predicted class. We retain the
                probabilities for the classes to enable validation of
                low-confidence pages.
              </div>
              <ul className="list-disc ml-6 space-y-4 text-muted-foreground">
                <li className="space-y-2">
                  <div>
                    <strong>Architecture:</strong> The Page Classifier Model
                    combines three layers:
                    <ol className="list-decimal space-y-1 pl-5">
                      <li>
                        An XGBoost model that scores each page{" "}
                        <em>individually</em>, using engineered text/HTML/layout
                        features
                      </li>
                      <li>
                        A BiLSTM + constrained CRF that refines those scores
                        into a coherent, document-level page sequence
                      </li>
                      <li>
                        A post-processing step that sets all pages after the
                        first high-confidence{" "}
                        <span className="font-mono text-sm text-foreground">
                          sig
                        </span>{" "}
                        block to{" "}
                        <span className="font-mono text-sm text-foreground">
                          back_matter
                        </span>
                      </li>
                    </ol>
                  </div>
                </li>
                <li className="space-y-2">
                  <div>
                    <strong>Training corpus:</strong> We train both models on an
                    80/20 random split of the full set of{" "}
                    <InfoDisclosure
                      isCoarsePointer={isCoarsePointer}
                      triggerClassName="cursor-help appearance-none bg-transparent p-0 text-inherit underline decoration-dotted underline-offset-4"
                      label={<strong>31,864</strong>}
                      content={
                        <>
                          Did we really label 31,864 pages by hand? Yes—but it
                          wasn't that bad. We had GPT build us a custom labeling
                          interface that allowed us to select all body pages in
                          a single go, so the whole process took less than three
                          hours.
                        </>
                      }
                    />{" "}
                    manually labeled pages, stratified (with distributional
                    matching) by length of backmatter section (bucketed into 4
                    buckets), overall length (bucketed into 4 buckets), and year
                    (bucketed into 5-year windows).
                  </div>
                </li>
              </ul>

              <p className="text-muted-foreground">
                Below are model performance metrics for the baseline XGB model,
                the BiLSTM + CRF model, and the BiLSTM + CRF with
                post-processing, all as run on a holdout set of 36 agreements.
                While the XGB model is somewhat middling on its own, taking into
                account and enforcing page order improves performance, bringing the F1 score up from{" "}
                <strong>{formatMetric(baselineClassifierF1)}</strong> to{" "}
                <strong>{formatMetric(postProcessingF1)}</strong>.{" "}
              </p>
              {metricsData ? (
                <Suspense
                  fallback={
                    <Card className="border-border/70 bg-card/70 p-5 text-sm text-muted-foreground">
                      Loading model metrics...
                    </Card>
                  }
                >
                  <LazyClassifierEvalMetrics data={metricsData.classifier} />
                </Suspense>
              ) : (
                <Card className="border-border/70 bg-card/70 p-5 text-sm text-muted-foreground">
                  Loading model metrics...
                </Card>
              )}
            </div>
            <div id="tagging-model" className="scroll-mt-32 pt-2 space-y-4">
              <h3 className="text-lg font-semibold text-foreground">
                Tagging Model
              </h3>
              <div className="text-muted-foreground">
                The Tagging Model is the heart of our pipeline. It takes as
                inputs plain-text body pages—as identified via the Page
                Classifier model—and outputs text with Article, Section, and
                Page entities tagged. We then concatenate all pages and use the
                tags to create XML. We persist into our database all spans where
                the Tagging Model's confidence fell below the threshold of 0.8;
                we feed those spans to an LLM for tag reconciliation.
              </div>
              <ul className="list-disc ml-6 space-y-4 text-muted-foreground">
                <li className="space-y-2">
                  <div>
                    <strong>Architecture:</strong> The Tagging Model fine-tunes{" "}
                    <a
                      href="https://huggingface.co/blog/modernbert"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary underline underline-offset-2 hover:underline"
                    >
                      <span className="font-mono text-sm text-foreground">
                        ModernBERT-base
                      </span>
                    </a>{" "}
                    for token-level labeling of Article, Section, and Page
                    entities. We use a BIOES scheme for Pages and a BIOE scheme
                    for Articles and Sections.
                  </div>
                </li>
                <li className="space-y-2">
                  <div>
                    <strong>Training corpus:</strong> We used{" "}
                    <span className="font-mono text-sm text-foreground">
                      gpt-5.1
                    </span>{" "}
                    to create a dataset of <strong>7,500</strong> labeled pages:{" "}
                    <strong>6,306</strong> come from a random set of{" "}
                    <strong>91</strong> complete agreements (
                    <span className="font-mono text-sm text-foreground">
                      body
                    </span>{" "}
                    pages only) and we selected an additional{" "}
                    <strong>1,194</strong> to{" "}
                    <InfoDisclosure
                      isCoarsePointer={isCoarsePointer}
                      triggerClassName="cursor-help appearance-none bg-transparent p-0 text-inherit underline decoration-dotted underline-offset-4"
                      label="upsample Article entities"
                      content={
                        <>
                          Because Article entities are relatively less common,
                          creating training data by ingesting complete
                          agreements—rather than sampling for specific kinds of
                          entities—produces a training corpus with fewer
                          examples of Article entities. So, after ingesting 91
                          complete agreements, we selected an additional 1,194
                          pages, filtering to those that mentioned either
                          "Article" or "ARTICLE".
                        </>
                      }
                    />
                    . We split this corpus into fixed sets for training (80%),
                    validation (10%), and testing (10%), reserving the test set
                    for the final evaluation only. To improve the model's
                    ability to identify Article entities—and also to ensure that
                    the distribution of entities in our validation and test sets
                    is representative of agreements—we send all 1,194 upsampled
                    pages to the training set. We then stratify the remaining
                    pages based on 5-year window, the presence of Article
                    entities, and the presence of Section entities.
                  </div>
                </li>
                <li className="space-y-2">
                  <div>
                    <strong>Experiment suite:</strong> Outside of model
                    hyperparameters, which we tune using Optuna, we have three
                    performance levers available: 1) the size of our training
                    data corpus, and the distribution of entity types within it;
                    2) the penalties we assign to incorrect predictions of
                    Articles, Sections, and Pages; and 3) which post-processing
                    transformations we perform. To evaluate the impact of each
                    of these on model performance—all experiments are evaluated
                    against our validation set—we first train a baseline model
                    to select and freeze hyperparameters, which we then reuse
                    across all experiments.
                  </div>
                </li>
              </ul>

              {metricsData ? (
                <Suspense
                  fallback={
                    <Card className="border-border/70 bg-card/70 p-5 text-sm text-muted-foreground">
                      Loading model metrics...
                    </Card>
                  }
                >
                  <LazyNerEvalMetrics
                    data={metricsData.ner}
                    showValidationBlocks={false}
                  />
                </Suspense>
              ) : (
                <Card className="border-border/70 bg-card/70 p-5 text-sm text-muted-foreground">
                  Loading model metrics...
                </Card>
              )}
            </div>
            <div
              id="taxonomy-model"
              className="scroll-mt-32 pt-2 space-y-4"
            >
              <h3 className="text-lg font-semibold text-foreground">
                Taxonomy Model
              </h3>
              <ComingSoon title="Mapping to the Pandects taxonomy" />
            </div>
          </section>

          <section id="gaps-and-callouts" className="scroll-mt-32 space-y-4">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Gaps and Other Call Outs
            </h2>
            <ul className="list-disc space-y-2 pl-6 text-muted-foreground">
              <li>
                We force each agreement into our five-class structure—
                <span className="font-mono text-sm text-foreground">
                  front_matter
                </span>
                , <span className="font-mono text-sm text-foreground">toc</span>
                ,{" "}
                <span className="font-mono text-sm text-foreground">body</span>,{" "}
                <span className="font-mono text-sm text-foreground">sig</span>,{" "}
                <span className="font-mono text-sm text-foreground">
                  back_matter
                </span>
                —but this structure is not the best fit for some agreements. For
                instance, some agreements place definition sections into
                appendices.
              </li>
              <li>
                At the moment, we do not process agreements that are not
                paginated. In our training set of 399 agreements, 31 are not
                paginated.
              </li>
              <li>
                The creators of the DMA Corpus started with merger data from
                FactSet and then filtered down SEC filings to the FactSet deals.
                We take a different approach: instead of identifying deals and
                matching them to filings, our Exhibit Model simply identifies
                filings that are sufficiently similar to bona fide filings,
                assumes they are from legitimate M&A deals, and then fills in
                deal metadata after the fact. This means that we might pull in
                some filings that are not properly M&A agreements.
              </li>
            </ul>
          </section>

          <section id="validations" className="scroll-mt-32 space-y-4">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Validations
            </h2>
            <p className="text-muted-foreground">
              Although the vast majority of agreements sail through our pipeline
              without issue, some do require manual validation. We have two
              validations in place:
            </p>
            <ol className="list-decimal ml-6 space-y-4 text-muted-foreground">
              <li className="space-y-2">
                <div>
                  <strong>Page label validation:</strong> We manually review
                  labels for agreements where either:
                </div>
                <ul className="list-disc space-y-1 pl-5">
                  <li>
                    At least one page label is out of order—for example, a{" "}
                    <span className="font-mono text-sm text-foreground">
                      back_matter
                    </span>{" "}
                    label before a{" "}
                    <span className="font-mono text-sm text-foreground">
                      sig
                    </span>{" "}
                    label.
                  </li>
                  <li>
                    At least one page has a prediction confidence between 30%
                    and 70%; where there is a high-confidence signature page (≥
                    95%), this condition applies only to pages prior to that
                    signature page.
                  </li>
                </ul>
              </li>
              <li className="space-y-2">
                <div>
                  <strong>Tag reconciliation:</strong> We manually review
                  uncertain spans when LLM rulings cannot be reconciled with
                  existing tags and we flag the page with a label error:
                </div>
                <ul className="list-disc space-y-1 pl-5">
                  <li>
                    A ruling intersects more than one tagged span or only
                    partially overlaps a span.
                  </li>
                  <li>
                    A ruling sits entirely within a tagged span but assigns a
                    different label.
                  </li>
                </ul>
              </li>
            </ol>
          </section>
        </div>
      </div>
    </PageShell>
  );
}
