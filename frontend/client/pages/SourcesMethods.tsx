import type { ReactNode } from "react";
import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import ResizeObserverPolyfill from "resize-observer-polyfill";
import { cn } from "@/lib/utils";
import { AdaptiveTooltip } from "@/components/ui/adaptive-tooltip";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
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


export default function SourcesMethods() {
  const [activeSection, setActiveSection] = useState("");
  const firstStepRef = useRef<HTMLDivElement | null>(null);
  const lastStepRef = useRef<HTMLDivElement | null>(null);
  const pipelineRef = useRef<HTMLDivElement | null>(null);
  const progressRef = useRef<HTMLDivElement | null>(null);
  const [pipelineLine, setPipelineLine] = useState({ top: 0, height: 0 });
  const lastScrollToRef = useRef(0);
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

    lastScrollToRef.current = Date.now();
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
        // Add will-change during animation
        progressRef.current.style.willChange = "transform";
        progressRef.current.style.transform = `scaleY(${clamped})`;
        // Remove will-change after animation completes
        requestAnimationFrame(() => {
          setTimeout(() => {
            if (progressRef.current) {
              progressRef.current.style.willChange = "auto";
            }
          }, 300);
        });
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
      if (Date.now() - lastScrollToRef.current > 500) {
        return;
      }
      if (
        [
          "exhibit-model",
          "page-classifier-model",
          "tagging-model",
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

  // Cleanup will-change on unmount
  useEffect(() => {
    return () => {
      if (progressRef.current) {
        progressRef.current.style.willChange = "auto";
      }
    };
  }, []);

  const validationClassifierF1 = 0.9666670514543612;
  const finalClassifierF1 = 0.9507914467270675;
  const formatMetric = (value: number) => `${(value * 100).toFixed(2)}%`;

  return (
    <PageShell
      size="xl"
      title="Sources & Methods"
    >
      <div className="grid gap-8 lg:grid-cols-[280px_1fr]">
        <aside className="hidden lg:block">
          <div className="sticky top-20">
            <Card className="border-border/60 bg-background/70 p-3 backdrop-blur">
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
                            : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
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

        <article className="space-y-12 min-w-0">
          <section id="overview" className="scroll-mt-32 space-y-4" aria-labelledby="overview-heading">
            <h2 id="overview-heading" className="text-2xl font-semibold tracking-tight text-foreground">
              Overview
            </h2>
            <p className="text-muted-foreground">
              Pandects sources agreements from the SEC's EDGAR database, then
              runs each agreement through a purpose-built pipeline that turns
              text and messy HTML into clean, taxonomized XML. Conceptually, our
              pipeline solves several distinct problems.
            </p>
            <div className="max-w-4xl rounded-2xl border border-border/60 bg-card/50 p-6 shadow-sm">
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
                        We parse through Exhibit 2 and Exhibit 10
                        filings at
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
                        Monthly Updates
                      </div>
                      <p className="text-sm text-foreground">
                        Our pipeline runs every month, ensuring the Pandects dataset
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
            aria-labelledby="data-pipeline-architecture-heading"
          >
            <h2 id="data-pipeline-architecture-heading" className="text-2xl font-semibold tracking-tight text-foreground">
              Pipeline Architecture
            </h2>
            <p className="max-w-3xl text-muted-foreground">
              Our pipeline takes raw EDGAR filings and produces clean XML,
              structured sections, and taxonomy labels. We use three ML
              models plus targeted LLM calls where they are the better tool. We
              use{" "}
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
                  className="absolute left-0 top-0 h-full w-full origin-top bg-gradient-to-b from-emerald-300 via-emerald-500 to-emerald-600 shadow-[0_0_10px_rgba(16,185,129,0.35)]"
                  style={{ transform: "scaleY(0)" }}
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
                        className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-foreground transition-colors hover:border-emerald-500/40"
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
                      <AdaptiveTooltip
                        trigger={
                          <button
                            type="button"
                            aria-label="Why we split and classify pages"
                            className="tooltip-help-trigger-compact"
                          >
                            ?
                          </button>
                        }
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
                        tooltipProps={{
                          side: "top",
                          className: "max-w-xs border-border/60 bg-background/95 text-xs text-foreground shadow-lg",
                        }}
                        delayDuration={300}
                        popoverProps={{
                          side: "top",
                          className: "max-w-xs border-border/60 bg-background/95 text-xs text-foreground shadow-lg",
                        }}
                      />
                      <a
                        href="#page-classifier-model"
                        className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-foreground transition-colors hover:border-emerald-500/40"
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
                            <span className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                              HTML
                            </span>
                          </div>
                          <p className="mt-2 text-xs font-mono text-muted-foreground">
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;p style="margin:0pt;text-align:center;..."&gt;
                            </span>
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;font style="..."&gt;
                            </span>
                            ARTICLE&nbsp;2
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;/font&gt;&lt;/p&gt;
                            </span>
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;p style="..."&gt;&lt;font style="..."&gt;
                            </span>
                            THE ARRANGEMENT
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;/font&gt;&lt;/p&gt;
                            </span>
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;p style="...text-indent:-72pt;..."&gt;&lt;font&gt;
                            </span>
                            Section&nbsp;2.1
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;font&gt;&lt;/font&gt;
                            </span>
                            Arrangement
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;/font&gt;&lt;/p&gt;
                            </span>
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;p style="...text-indent:36pt;..."&gt;&lt;font&gt;
                            </span>
                            The Company, the Parent and the Purchaser agree that
                            the Arrangement will be implemented…
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;/font&gt;&lt;/p&gt;
                            </span>
                          </p>
                        </div>
                        <div>
                          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            After
                            <span className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                              Plain Text
                            </span>
                          </div>
                          <p className="mt-2 text-sm text-foreground">
                            ARTICLE 2<br />
                            <br />
                            THE ARRANGEMENT<br />
                            <br />
                            Section 2.1 Arrangement<br />
                            <br />
                            The Company, the Parent and the Purchaser agree that
                            the Arrangement will be implemented…
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
                        className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-foreground transition-colors hover:border-emerald-500/40"
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
                            <span className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                              Plain Text
                            </span>
                          </div>
                          <p className="mt-2 text-sm text-muted-foreground">
                            ARTICLE 2<br />
                            <br />
                            THE ARRANGEMENT<br />
                            <br />
                            Section 2.1 Arrangement<br />
                            <br />
                            The Company, the Parent and the Purchaser agree that
                            the Arrangement will be implemented…
                          </p>
                        </div>
                        <div>
                          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            After
                            <span className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                              Tagged Text
                            </span>
                          </div>
                          <p className="mt-2 text-xs font-mono text-foreground">
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;article&gt;
                            </span>
                            ARTICLE 2<br />
                            <br />
                            THE ARRANGEMENT
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;/article&gt;
                            </span>
                            <br />
                            <br />
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;section&gt;
                            </span>
                            Section 2.1 Arrangement
                            <span className="rounded-sm bg-amber-100/80 px-1 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100">
                              &lt;/section&gt;
                            </span>
                            <br />
                            <br />
                            The Company, the Parent and the Purchaser agree that
                            the Arrangement will be implemented…
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
                  <div className="space-y-3">
                    <div className="font-mono text-sm font-semibold text-foreground">
                      XML Generation
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Converts tagged body pages into agreement XML and upserts
                      section records into the sections table.
                    </p>
                  </div>
                </li>
                <li className="relative flex gap-4">
                  <div className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70">
                    5
                  </div>
                  <div className="space-y-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-foreground">
                        XML Repair Loop
                      </span>
                      <span className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-foreground">
                        LLM
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      When XML validation fails, we send the affected pages to
                      an LLM for repair, reconcile the fixes back into the
                      tagged text, regenerate XML, and repeat until the
                      agreement either validates or is exhausted by the repair
                      pass budget.
                    </p>
                  </div>
                </li>
                <li className="relative flex gap-4">
                  <div className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70">
                    6
                  </div>
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-foreground">
                        Taxonomy
                      </span>
                      <span className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-foreground">
                        LLM
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Sends section and nearby heading context to an LLM, then
                      assigns section-level taxonomy labels for cross-agreement
                      analysis.
                    </p>
                  </div>
                </li>
                <li className="relative flex gap-4">
                  <div
                    ref={lastStepRef}
                    className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-xs font-semibold text-muted-foreground before:absolute before:left-full before:top-1/2 before:h-px before:w-3 before:-translate-y-1/2 before:bg-border/70"
                  >
                    7
                  </div>
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-foreground">
                        Metadata
                      </span>
                      <span className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold text-foreground">
                        LLM
                      </span>
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
            aria-labelledby="ml-models-heading"
          >
            <h2 id="ml-models-heading" className="text-2xl font-semibold tracking-tight text-foreground">
              ML Models
            </h2>
            <div className="text-muted-foreground">
              While we'd prefer to outsource all inference tasks to an LLM,
              Pandects is a small-scale operation run on a shoestring budget,
              meaning we don't have the resources to send hundreds of thousands
              of pages of agreements to proprietary LLMs—and as of October 2025,{" "}
              <AdaptiveTooltip
                trigger={
                  <button
                    type="button"
                    className="cursor-help appearance-none bg-transparent p-0 text-inherit underline decoration-dotted underline-offset-4"
                  >
                    open-source models
                  </button>
                }
                content={
                  <>
                    We tried DeepSeek R1 on NYU's HPC clusters, and the results
                    were less than reliable.
                  </>
                }
                tooltipProps={{
                  side: "top",
                  className: "max-w-xs border-border/60 bg-background/95 text-xs text-foreground shadow-lg",
                }}
                delayDuration={300}
                popoverProps={{
                  side: "top",
                  className: "max-w-xs border-border/60 bg-background/95 text-xs text-foreground shadow-lg",
                }}
              />{" "}
              still made too many mistakes for us to be comfortable using them
              at scale. Our solution: use the latest and greatest LLMs to
              generate high-quality training data from a limited sample of
              agreements and then train up more routine machine learning models
              to do the work we'd otherwise outsource to LLMs.
            </div>
            <p className="text-muted-foreground">
              This section describes the three ML models at the heart of our data
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
              <h3 className="text-xl font-semibold text-foreground">
                Exhibit Model
              </h3>
              <div className="text-muted-foreground">
                The Exhibit Model takes as inputs filings from the SEC's EDGAR
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
                scrape each filing's index page to extract Exhibit 10.* and
                2.* links, fetch and render the exhibit content as text,
                and use the Exhibit Model to classify candidates.
              </div>
              <ul className="list-disc ml-6 space-y-4 text-muted-foreground">
                <li className="space-y-2">
                  <div>
                    <strong>Architecture:</strong> The Exhibit Model is a tuned
                    logistic-regression classifier over three feature blocks:
                    (1) <em>document-level features</em> such as opening title
                    phrases, hard-negative phrase counts, document length, and
                    M&A keyword density; (2) <em>hashed word- and
                    character-level TF-IDF features</em>; and (3){" "}
                    <em>similarity features</em> capturing max, mean, and
                    median cosine similarity to the training agreement corpus.
                    We also apply a minimum-length hard-negative rule and use a
                    tuned decision threshold of{" "}
                    <span className="font-mono text-sm text-foreground">
                      0.96
                    </span>
                    .
                  </div>
                </li>
                <li className="space-y-2">
                  <div>
                    <strong>Training corpus:</strong> We begin with{" "}
                    <strong>8,889</strong> labeled exhibits:{" "}
                    <strong>7,919</strong> positives and{" "}
                    <strong>970</strong> negatives. After dropping{" "}
                    <strong>23</strong> short positive exhibits to align the{" "}
                    training set with the model&apos;s minimum-length{" "}
                    hard-negative rule, we split the remaining{" "}
                    <strong>8,866</strong> exhibits with a stable manifest into{" "}
                    <strong>6,206</strong> training exhibits,{" "}
                    <strong>886</strong> validation exhibits, and{" "}
                    <strong>1,774</strong> holdout test exhibits.
                  </div>
                </li>
              </ul>
              {metricsData ? (
                <Suspense
                  fallback={
                    <Card className="border-border/60 bg-card/70 p-6">
                      <div className="space-y-4">
                        <Skeleton className="h-6 w-48" />
                        <Skeleton className="h-64 w-full" />
                        <div className="grid grid-cols-2 gap-4">
                          <Skeleton className="h-12 w-full" />
                          <Skeleton className="h-12 w-full" />
                        </div>
                      </div>
                    </Card>
                  }
                >
                  <LazyExhibitEvalMetrics data={metricsData.exhibit} />
                </Suspense>
              ) : (
                <Card className="border-border/60 bg-card/70 p-6">
                  <div className="space-y-4">
                    <Skeleton className="h-6 w-48" />
                    <Skeleton className="h-64 w-full" />
                    <div className="grid grid-cols-2 gap-4">
                      <Skeleton className="h-12 w-full" />
                      <Skeleton className="h-12 w-full" />
                    </div>
                  </div>
                </Card>
              )}
            </div>
            <div
              id="page-classifier-model"
              className="scroll-mt-32 pt-2 space-y-4"
            >
              <h3 className="text-xl font-semibold text-foreground">
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
                    is a document-level monotonic CRF:
                    <ol className="list-decimal space-y-1 pl-5">
                      <li>
                        Each page is featurized with positional, structural,
                        and lexical signals, including relative page position,
                        heading patterns, TOC dots, signature and witness
                        markers, annex and appendix anchors, and compact TF-IDF
                        word features.
                      </li>
                      <li>
                        A constrained CRF predicts the full page sequence in
                        one pass while enforcing monotonic transitions through{" "}
                        <span className="font-mono text-sm text-foreground">
                          front_matter
                        </span>
                        ,{" "}
                        <span className="font-mono text-sm text-foreground">
                          toc
                        </span>
                        ,{" "}
                        <span className="font-mono text-sm text-foreground">
                          body
                        </span>
                        ,{" "}
                        <span className="font-mono text-sm text-foreground">
                          sig
                        </span>
                        , and{" "}
                        <span className="font-mono text-sm text-foreground">
                          back_matter
                        </span>
                        .
                      </li>
                    </ol>
                  </div>
                </li>
                <li className="space-y-2">
                  <div>
                    <strong>Training corpus:</strong> We train the page CRF on
                    the full set of{" "}
                    <AdaptiveTooltip
                      trigger={
                        <button
                          type="button"
                          className="cursor-help appearance-none bg-transparent p-0 text-inherit underline decoration-dotted underline-offset-4"
                        >
                          <strong>67,596</strong>
                        </button>
                      }
                      content={
                        <>
                          Did we really label 67,596 pages by hand? Yes—but it
                          wasn't that bad. We had GPT build us a custom labeling
                          interface that allowed us to select all body pages in
                          a single go, so adding agreements was much faster
                          than labeling every page one by one.
                        </>
                      }
                      tooltipProps={{
                        side: "top",
                        className: "max-w-xs border-border/60 bg-background/95 text-xs text-foreground shadow-lg",
                      }}
                      delayDuration={300}
                      popoverProps={{
                        side: "top",
                        className: "max-w-xs border-border/60 bg-background/95 text-xs text-foreground shadow-lg",
                      }}
                    />{" "}
                    manually labeled pages across 673 agreements, grouped by
                    agreement and split with a stable manifest stratified by
                    announcement year and agreement type.
                  </div>
                </li>
              </ul>

              <p className="text-muted-foreground">
                Below are model performance metrics for the tuned validation CRF
                run and the final holdout test run. The model reaches a macro
                F1 of <strong>{formatMetric(validationClassifierF1)}</strong> on
                validation and <strong>{formatMetric(finalClassifierF1)}</strong>{" "}
                on the final test split, with most errors concentrated
                at the body/back-matter boundary.
              </p>
              {metricsData ? (
                <Suspense
                  fallback={
                    <Card className="border-border/60 bg-card/70 p-6">
                      <div className="space-y-4">
                        <Skeleton className="h-6 w-48" />
                        <Skeleton className="h-64 w-full" />
                        <div className="grid grid-cols-3 gap-4">
                          <Skeleton className="h-12 w-full" />
                          <Skeleton className="h-12 w-full" />
                          <Skeleton className="h-12 w-full" />
                        </div>
                      </div>
                    </Card>
                  }
                >
                  <LazyClassifierEvalMetrics data={metricsData.classifier} />
                </Suspense>
              ) : (
                <Card className="border-border/60 bg-card/70 p-6">
                  <div className="space-y-4">
                    <Skeleton className="h-6 w-48" />
                    <Skeleton className="h-64 w-full" />
                    <div className="grid grid-cols-3 gap-4">
                      <Skeleton className="h-12 w-full" />
                      <Skeleton className="h-12 w-full" />
                      <Skeleton className="h-12 w-full" />
                    </div>
                  </div>
                </Card>
              )}
            </div>
            <div id="tagging-model" className="scroll-mt-32 pt-2 space-y-4">
              <h3 className="text-xl font-semibold text-foreground">
                Tagging Model
              </h3>
              <div className="text-muted-foreground">
                The Tagging Model is the heart of our pipeline. It takes as
                inputs plain-text body pages—as identified via the Page
                Classifier model—and outputs text with Article, Section, and
                Page entities tagged. We then concatenate all pages and use the
                tags to create XML. We also persist low-confidence token spans
                for downstream review and AI-assisted reconciliation.
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
                    entities. The current production recipe uses an independent
                    decoder, a lightweight auxiliary boundary head for Article
                    and Section start/end detection, focal token loss,
                    preserved input casing, and boundary-mix sampling over
                    fixed-length token windows. At inference time, overlapping
                    windows are stitched by averaging logits and then lightly
                    repaired into legal BIO-style sequences before tags are
                    rendered back into the page text. We use BIOE tags for
                    Articles and Sections, plus BIOES-style tags for Pages.
                  </div>
                </li>
                <li className="space-y-2">
                  <div>
                    <strong>Training corpus:</strong> We used{" "}
                    <span className="font-mono text-sm text-foreground">
                      gpt-5.1
                    </span>{" "}
                    to create a page-level structural tagging corpus of{" "}
                    <strong>15,000</strong> agreement pages (
                    <span className="font-mono text-sm text-foreground">
                      body
                    </span>{" "}
                    pages only) drawn from <strong>395</strong> agreements. Of
                    those, <strong>10,016</strong> pages
                    contain at least one structural tag and{" "}
                    <strong>4,984</strong> are clean negatives. The corpus
                    includes <strong>3,130</strong> pages with Article tags and{" "}
                    <strong>9,990</strong> pages with Section tags. We also
                    designated <strong>3,147</strong> pages to{" "}
                    <AdaptiveTooltip
                      trigger={
                        <button
                          type="button"
                          className="cursor-help appearance-none bg-transparent p-0 text-inherit underline decoration-dotted underline-offset-4"
                        >
                          upsample Article entities
                        </button>
                      }
                      content={
                        <>
                          Article spans are structurally rarer and more brittle
                          at the exact end boundary than Section spans. We
                          therefore marked 3,147 Article-heavy pages for
                          targeted sampling in training so the model saw more
                          of those heading patterns.
                        </>
                      }
                      tooltipProps={{
                        side: "top",
                        className: "max-w-xs border-border/60 bg-background/95 text-xs text-foreground shadow-lg",
                      }}
                      delayDuration={300}
                      popoverProps={{
                        side: "top",
                        className: "max-w-xs border-border/60 bg-background/95 text-xs text-foreground shadow-lg",
                      }}
                    />
                    . We split the corpus by <strong>agreement</strong>, not by
                    page, so that pages from the same agreement never land in
                    both train and holdout sets. The fixed manifests contain{" "}
                    <strong>11,794</strong> train pages across{" "}
                    <strong>348</strong> agreements, <strong>1,508</strong>{" "}
                    validation pages across <strong>23</strong> agreements, and{" "}
                    <strong>1,698</strong> test pages across{" "}
                    <strong>24</strong> agreements; the test split is reserved
                    for final evaluation only. All 3,147 Article-oversampled
                    pages remain train-only so model comparisons run on the
                    same agreement-level partitions without leakage.
                  </div>
                </li>
              </ul>

              {metricsData ? (
                <Suspense
                  fallback={
                    <Card className="border-border/60 bg-card/70 p-6">
                      <div className="space-y-4">
                        <Skeleton className="h-6 w-48" />
                        <Skeleton className="h-64 w-full" />
                        <div className="grid grid-cols-3 gap-4">
                          <Skeleton className="h-12 w-full" />
                          <Skeleton className="h-12 w-full" />
                          <Skeleton className="h-12 w-full" />
                        </div>
                      </div>
                    </Card>
                  }
                >
                  <LazyNerEvalMetrics
                    data={metricsData.ner}
                  />
                </Suspense>
              ) : (
                <Card className="border-border/60 bg-card/70 p-6">
                  <div className="space-y-4">
                    <Skeleton className="h-6 w-48" />
                    <Skeleton className="h-64 w-full" />
                    <div className="grid grid-cols-3 gap-4">
                      <Skeleton className="h-12 w-full" />
                      <Skeleton className="h-12 w-full" />
                      <Skeleton className="h-12 w-full" />
                    </div>
                  </div>
                </Card>
              )}
            </div>
          </section>

          <section id="gaps-and-callouts" className="scroll-mt-32 space-y-4" aria-labelledby="gaps-and-callouts-heading">
            <h2 id="gaps-and-callouts-heading" className="text-2xl font-semibold tracking-tight text-foreground">
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
              <li>
                We apply our taxonomy at the <em>section</em> level, meaning
                that some sections may belong to multiple taxonomic categories.
                For instance, sections that cover governing law and forum
                selection in one pass will belong to both{" "}
                <span className="font-mono text-sm text-foreground">
                  Governing Law
                </span>{" "}
                and{" "}
                <span className="font-mono text-sm text-foreground">
                  Forum Selection / Venue; Submission to Jurisdiction
                </span>
                . In the future, to complement section-level classification, we
                plan to create a clause-level taxonomy as well.
              </li>
              <li>
                We currently assign sections to taxonomic classes by sending the target
                section title and the title of its parent article, as well as
                section and article titles from both preceding and following
                sections, to{" "}
                <span className="font-mono text-sm text-foreground">
                  gpt-5.1
                </span>{" "}
                and{" "}
                <span className="font-mono text-sm text-foreground">
                  gpt-5-mini
                </span>{" "}
                (as our free token allowance permits). This means taxonomy is
                currently assigned by LLM using titles and nearby structural
                context alone, with no attention to section body text.
              </li>
            </ul>
          </section>

          <section id="validations" className="scroll-mt-32 space-y-4" aria-labelledby="validations-heading">
            <h2 id="validations-heading" className="text-2xl font-semibold tracking-tight text-foreground">
              Validations
            </h2>
            <p className="text-muted-foreground">
              Validation happens at several points in the pipeline rather than
              in a single end-of-run review step. In practice, we use four
              checks:
            </p>
            <ol className="list-decimal ml-6 space-y-4 text-muted-foreground">
              <li className="space-y-2">
                <div>
                  <strong>Exhibit review:</strong> Before we treat an agreement
                  as confirmed, we manually review every agreement surfaced by
                  the Exhibit Model, except that non-amended agreements titled
                  "Agreement and Plan of Merger", "Business Combination
                  Agreement" or "Membership Interest Purchase Agreement" are
                  assumed valid by default.
                </div>
              </li>
              <li className="space-y-2">
                <div>
                  <strong>Page classification review:</strong> During
                  pre-processing, a dedicated agreement-level review model uses
                  page-classification uncertainty and sequence-risk signals to
                  flag agreements whose page labels should be reviewed before
                  they move downstream.
                </div>
              </li>
              <li className="space-y-2">
                <div>
                  <strong>XML verification:</strong> Fresh XML first goes
                  through strict structural validation with hard rules around
                  parseability, body structure, article ordering, and section
                  numbering. XML that survives those rule checks is then sent to
                  an LLM verifier, which marks the document as verified,
                  invalid, or unresolved.
                </div>
              </li>
              <li className="space-y-2">
                <div>
                  <strong>AI repair for failed XML:</strong> When XML fails
                  validation, we route the agreement into the repair cycle and
                  send the specific pages tied to the XML failure reasons to an
                  LLM for full-page repair. Those repairs are reconciled back
                  into the tagged text, XML is rebuilt, and the rebuilt XML is
                  verified again.
                </div>
              </li>
            </ol>
          </section>
        </article>
      </div>
    </PageShell>
  );
}
