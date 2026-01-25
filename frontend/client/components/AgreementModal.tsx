import { useEffect, useState, useRef, useId } from "react";
import { createPortal } from "react-dom";
import {
  X,
  ArrowLeft,
  PanelLeftOpen,
  PanelLeftClose,
  ExternalLink,
  ChevronDown,
} from "lucide-react";
import { FlagAsInaccurateButton } from "@/components/FlagAsInaccurateButton";
import { cn } from "@/lib/utils";
import { useAgreement } from "@/hooks/use-agreement";
import { XMLRenderer } from "./XMLRenderer";
import { TableOfContents } from "./TableOfContents";
import { useIsMobile } from "@/hooks/use-mobile";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import {
  animateScrollTop,
  getScrollTopForElementInContainer,
} from "@/lib/scroll";
import {
  formatDateValue,
  formatNumberValue,
  formatCurrencyValue,
  formatEnumValue,
  formatTextValue,
  formatBooleanValue,
} from "@/lib/format-utils";

interface AgreementModalProps {
  isOpen: boolean;
  onClose: () => void;
  agreementUuid: string;
  targetSectionUuid?: string;
  agreementMetadata?: {
    year: string;
    target: string;
    acquirer: string;
  };
}

export function AgreementModal({
  isOpen,
  onClose,
  agreementUuid,
  targetSectionUuid,
  agreementMetadata,
}: AgreementModalProps) {
  const { agreement, isLoading, error, fetchAgreement, clearAgreement } =
    useAgreement();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isTocOpen, setIsTocOpen] = useState(false);
  const [highlightedSection, setHighlightedSection] = useState<string | null>(
    null,
  );
  const contentRef = useRef<HTMLDivElement>(null);
  const cancelScrollAnimationRef = useRef<null | (() => void)>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const lastFocusedElementRef = useRef<HTMLElement | null>(null);
  const modalTitleId = useId();
  const modalDescriptionId = useId();
  const isMobile = useIsMobile();
  const canUseDOM = typeof document !== "undefined";
  const year = agreementMetadata?.year ?? agreement?.year;
  const target = agreementMetadata?.target ?? agreement?.target;
  const acquirer = agreementMetadata?.acquirer ?? agreement?.acquirer;
  const yearDisplay =
    year !== null && year !== undefined && year !== "" ? String(year) : null;
  const modalTitle =
    [yearDisplay, target, acquirer].filter(Boolean).join(" - ") ||
    "Agreement details";
  const mobileMetadataSummary = (() => {
    const parts: string[] = [];
    if (yearDisplay) parts.push(yearDisplay);
    const counterparties =
      target && acquirer ? `${target} → ${acquirer}` : target ?? acquirer;
    if (counterparties) parts.push(counterparties);
    return parts.join(" · ");
  })();
  const metadataSections = [
    {
      title: "Parties",
      items: [
        { label: "Target", value: formatTextValue(target) },
        { label: "Acquirer", value: formatTextValue(acquirer) },
        { label: "Target type", value: formatEnumValue(agreement?.target_type) },
        {
          label: "Acquirer type",
          value: formatEnumValue(agreement?.acquirer_type),
        },
        {
          label: "Target industry",
          value: formatTextValue(agreement?.target_industry),
        },
        {
          label: "Acquirer industry",
          value: formatTextValue(agreement?.acquirer_industry),
        },
        {
          label: "Target private equity",
          value: formatBooleanValue(agreement?.target_pe),
        },
        {
          label: "Acquirer private equity",
          value: formatBooleanValue(agreement?.acquirer_pe),
        },
      ],
    },
    {
      title: "Filing",
      items: [
        {
          label: "Filing date",
          value: formatDateValue(agreement?.filing_date),
        },
        {
          label: "Filing probability",
          value: formatNumberValue(agreement?.prob_filing),
        },
        {
          label: "Filing company name",
          value: formatTextValue(agreement?.filing_company_name),
        },
        {
          label: "Filing company CIK",
          value: formatTextValue(agreement?.filing_company_cik),
        },
        { label: "Form type", value: formatTextValue(agreement?.form_type) },
        {
          label: "Exhibit type",
          value: formatTextValue(agreement?.exhibit_type),
        },
      ],
    },
    {
      title: "Transaction",
      items: [
        {
          label: "Transaction price total",
          value: formatCurrencyValue(agreement?.transaction_price_total),
        },
        {
          label: "Transaction price stock",
          value: formatCurrencyValue(agreement?.transaction_price_stock),
        },
        {
          label: "Transaction price cash",
          value: formatCurrencyValue(agreement?.transaction_price_cash),
        },
        {
          label: "Transaction price assets",
          value: formatCurrencyValue(agreement?.transaction_price_assets),
        },
        {
          label: "Consideration",
          value: formatEnumValue(agreement?.transaction_consideration),
        },
        {
          label: "Deal type",
          value: formatEnumValue(agreement?.deal_type),
        },
        {
          label: "Purpose",
          value: formatEnumValue(agreement?.purpose),
        },
        {
          label: "Attitude",
          value: formatEnumValue(agreement?.attitude),
        },
        {
          label: "Deal status",
          value: formatEnumValue(agreement?.deal_status),
        },
      ],
    },
    {
      title: "Timeline",
      items: [
        {
          label: "Announce date",
          value: formatDateValue(agreement?.announce_date),
        },
        {
          label: "Close date",
          value: formatDateValue(agreement?.close_date),
        },
      ],
    },
  ];

  useEffect(() => {
    if (isOpen && agreementUuid) {
      fetchAgreement(agreementUuid, targetSectionUuid);
    } else if (!isOpen) {
      clearAgreement();
    }
  }, [isOpen, agreementUuid, targetSectionUuid, fetchAgreement, clearAgreement]);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [isOpen, onClose]);

  useEffect(() => {
    if (!isOpen) return;

    lastFocusedElementRef.current = document.activeElement as HTMLElement | null;

    const frame = window.requestAnimationFrame(() => {
      closeButtonRef.current?.focus();
    });

    return () => {
      window.cancelAnimationFrame(frame);
      lastFocusedElementRef.current?.focus?.();
      lastFocusedElementRef.current = null;
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !canUseDOM) return;
    const body = document.body;
    const currentCount = Number(body.dataset.modalCount ?? "0");
    const nextCount = currentCount + 1;
    body.dataset.modalCount = String(nextCount);
    const root = document.getElementById("root");
    if (root) {
      root.setAttribute("aria-hidden", "true");
      root.setAttribute("inert", "");
    }
    return () => {
      const updatedCount = Number(body.dataset.modalCount ?? "1") - 1;
      const safeCount = Math.max(0, updatedCount);
      body.dataset.modalCount = String(safeCount);
      if (safeCount === 0 && root) {
        root.removeAttribute("aria-hidden");
        root.removeAttribute("inert");
      }
    };
  }, [isOpen, canUseDOM]);

  const handleTrapFocus = (event: React.KeyboardEvent) => {
    if (event.key !== "Tab") return;
    const container = modalRef.current;
    if (!container) return;

    const focusable = Array.from(
      container.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])',
      ),
    ).filter((el) => el.getClientRects().length > 0);

    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement as HTMLElement | null;

    if (event.shiftKey && active === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const scrollToSection = (
    sectionUuid: string,
    shouldHighlight: boolean = false,
  ) => {
    if (!contentRef.current) return;

    // Find element with data-section-uuid attribute
    const sectionElement = contentRef.current.querySelector(
      `[data-section-uuid="${sectionUuid}"]`,
    ) as HTMLElement;

    if (sectionElement) {
      // First, check if this section/article is collapsed and expand it if needed
      const collapseButton = sectionElement.querySelector(
        'button[data-collapse-toggle="true"]',
      ) as HTMLButtonElement;
      if (collapseButton) {
        // Check if it's collapsed by looking for the content div that should be visible when expanded
        // In the XMLRenderer, collapsed content has children that are hidden/not rendered
        const contentContainer =
          sectionElement.querySelector(".agreement-children");
        const isCollapsed =
          !contentContainer || contentContainer.children.length === 0;

        if (isCollapsed) {
          // Section is collapsed, expand it first
          collapseButton.click();

          // Wait a moment for the expansion animation to complete before scrolling
          setTimeout(() => {
            performScroll();
          }, 150);
        } else {
          // Already expanded, scroll immediately
          performScroll();
        }
      } else {
        // No collapse button found, scroll immediately
        performScroll();
      }

      function performScroll() {
        const container = contentRef.current;
        if (!container) return;

        const targetScrollTop = getScrollTopForElementInContainer(
          container,
          sectionElement,
          { offsetPx: 8 },
        );

        cancelScrollAnimationRef.current?.();
        cancelScrollAnimationRef.current = animateScrollTop(
          container,
          targetScrollTop,
        );

        // Add highlighting if requested
        if (shouldHighlight) {
          setHighlightedSection(sectionUuid);
          // Remove highlight after animation
          setTimeout(() => {
            setHighlightedSection(null);
          }, 2000);
        }
      }
    }
  };

  useEffect(() => {
    if (agreement && targetSectionUuid) {
      // Very short delay to allow DOM to update, then immediate scroll
      const timer = setTimeout(() => {
        scrollToSection(targetSectionUuid, true);
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [agreement, targetSectionUuid]);

  if (!isOpen || !canUseDOM) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur supports-[backdrop-filter]:backdrop-blur-[1px] flex items-center justify-center p-0 sm:p-4"
      onClick={onClose}
    >
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={modalTitleId}
        aria-describedby={modalDescriptionId}
        className={cn(
          "bg-card text-foreground shadow-xl w-full flex flex-col overflow-hidden",
          // Mobile: full-screen modal for a native-app feel
          "h-[100dvh] max-h-[100dvh] rounded-none",
          // Desktop: centered, rounded
          "sm:h-full sm:max-h-[95vh] sm:rounded-lg sm:max-w-7xl",
        )}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleTrapFocus}
      >
        <h2 id={modalTitleId} className="sr-only">
          {modalTitle}
        </h2>
        <p id={modalDescriptionId} className="sr-only">
          Agreement document viewer
        </p>

        {/* Header */}
        <div className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/70 supports-[backdrop-filter]:backdrop-blur">
          <div className="flex items-center justify-between gap-3 px-3 py-3 sm:p-4">
            <div className="flex items-center gap-2 sm:gap-4 min-w-0">
              <Button
                onClick={onClose}
                variant="ghost"
                size="sm"
                className="gap-2 text-muted-foreground hover:text-foreground"
              >
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                <span className="hidden sm:inline">Back to Search</span>
                <span className="sm:hidden">Back</span>
              </Button>

              {/* Desktop: inline TOC toggle. Mobile: TOC lives in a sheet. */}
              {!isMobile && (
                <Button
                  onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                  variant="ghost"
                  size="sm"
                  className="gap-2 text-muted-foreground hover:text-foreground"
                >
                  {sidebarCollapsed ? (
                    <PanelLeftOpen className="h-4 w-4" aria-hidden="true" />
                  ) : (
                    <PanelLeftClose className="h-4 w-4" aria-hidden="true" />
                  )}
                  <span className="hidden sm:inline">
                    {sidebarCollapsed ? "Show" : "Hide"} Table of Contents
                  </span>
                  <span className="sm:hidden">Contents</span>
                </Button>
              )}

              {isMobile && (
                <Sheet open={isTocOpen} onOpenChange={setIsTocOpen}>
                  <SheetTrigger asChild>
                    <Button variant="outline" size="sm" className="gap-2">
                      <PanelLeftOpen className="h-4 w-4" aria-hidden="true" />
                      Contents
                    </Button>
                  </SheetTrigger>
                  <SheetContent
                    side="left"
                    className="w-[min(340px,100vw)] max-w-full p-0"
                  >
                    <SheetTitle className="sr-only">
                      Agreement contents
                    </SheetTitle>
                    <SheetDescription className="sr-only">
                      Browse agreement sections and jump to a clause.
                    </SheetDescription>
                    {agreement ? (
                      <TableOfContents
                        xmlContent={agreement.xml}
                        targetSectionUuid={targetSectionUuid}
                        onSectionClick={(uuid) => {
                          setIsTocOpen(false);
                          // Give the sheet a beat to start closing so scroll feels stable
                          setTimeout(() => scrollToSection(uuid, false), 50);
                        }}
                        className="h-[100dvh]"
                      />
                    ) : (
                      <div className="p-4 text-sm text-muted-foreground">
                        {isLoading ? "Loading…" : "Contents unavailable."}
                      </div>
                    )}
                  </SheetContent>
                </Sheet>
              )}
            </div>

            <div className="flex items-center gap-1">
              <FlagAsInaccurateButton
                source="agreement_view"
                agreementUuid={agreementUuid}
                className="shrink-0"
                tooltipSide="left"
              />
              <Button
                ref={closeButtonRef}
                onClick={onClose}
                variant="ghost"
                size="icon"
                className="h-10 w-10 text-muted-foreground hover:text-foreground"
                aria-label="Close"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </Button>
            </div>
          </div>
        </div>

        {/* Agreement Metadata */}
        {(agreementMetadata || agreement) && (
          <div className="border-b border-border bg-muted/40 px-4 py-3 sm:px-6 sm:py-4">
            {/* Key metadata always visible */}
            <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
              {yearDisplay && (
                <span className="inline-flex items-center rounded-full bg-background px-2 py-0.5 text-xs font-medium text-foreground ring-1 ring-border">
                  {yearDisplay}
                </span>
              )}
              {target && (
                <span className="text-foreground">
                  <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Target:</span>{" "}
                  <span className="font-medium">{target}</span>
                </span>
              )}
              {acquirer && (
                <span className="text-foreground">
                  <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Acquirer:</span>{" "}
                  <span className="font-medium">{acquirer}</span>
                </span>
              )}
              {agreement?.transaction_price_total && (
                <span className="text-foreground">
                  <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Price:</span>{" "}
                  <span className="font-medium">{formatCurrencyValue(agreement.transaction_price_total)}</span>
                </span>
              )}
              {agreement?.url && (
                <a
                  href={agreement.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Original Filing (opens in a new tab)"
                  className="inline-flex items-center gap-2 rounded-md px-2 py-1 text-xs font-medium text-primary transition-colors hover:bg-primary/10 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:ml-auto"
                  title="View original SEC filing"
                >
                  <ExternalLink
                    className="w-3.5 h-3.5 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform"
                    aria-hidden="true"
                  />
                  <span>Original Filing</span>
                </a>
              )}
            </div>
            <details className="group">
              <summary className="flex items-center justify-between gap-3 cursor-pointer select-none">
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium text-foreground">
                    Click to view deal metadata
                  </div>
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="group-open:hidden">Expand metadata</span>
                  <span className="hidden group-open:inline">Collapse metadata</span>
                  <ChevronDown
                    className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180"
                    aria-hidden="true"
                  />
                </div>
              </summary>

              <div className="mt-3 hidden sm:grid grid-cols-1 gap-3 lg:grid-cols-2">
                {metadataSections.map((section) => (
                  <section
                    key={section.title}
                    className="rounded-md border border-border/60 bg-background/70 p-3"
                  >
                    <h3 className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                      {section.title}
                    </h3>
                    <dl className="mt-2 grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
                      {section.items.map((item) => (
                        <div key={item.label} className="space-y-0.5">
                          <dt className="text-[11px] font-medium text-muted-foreground">
                            {item.label}
                          </dt>
                          <dd className="text-xs text-foreground break-words">
                            {item.value}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  </section>
                ))}
              </div>
              <div className="mt-3 rounded-md border border-dashed border-border/60 bg-background/70 px-3 py-2 text-xs text-muted-foreground sm:hidden">
                To see deal metadata, view on desktop.
              </div>
            </details>
          </div>
        )}

        {/* Content */}
        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Sidebar */}
          {!isMobile && !sidebarCollapsed && (
            <div className="w-80 border-r border-border bg-muted/30 flex-shrink-0">
              {agreement && (
                <TableOfContents
                  xmlContent={agreement.xml}
                  targetSectionUuid={targetSectionUuid}
                  onSectionClick={(uuid) => scrollToSection(uuid, false)}
                  className="h-full"
                />
              )}
            </div>
          )}

          {/* Main Content */}
          <div className="flex-1 min-w-0 overflow-hidden">
            {isLoading && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center" role="status" aria-live="polite">
                  <LoadingSpinner size="lg" aria-label="Loading agreement" className="mx-auto mb-4" />
                  <p className="text-muted-foreground">Loading agreement...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-destructive" role="alert">
                  <p className="mb-2">Failed to load agreement</p>
                  <p className="text-sm text-muted-foreground">{error}</p>
                </div>
              </div>
            )}

            {agreement && (
              <div
                ref={contentRef}
                className="h-full overflow-y-auto px-4 py-4 sm:p-6"
                style={{
                  scrollbarWidth: "thin",
                  scrollbarColor:
                    "hsl(var(--border)) hsl(var(--background))",
                }}
              >
                <div className="max-w-4xl mx-auto min-w-0">
                  {agreement.isRedacted ? (
                    <div className="mb-4">
                      <Alert>
                        <AlertTitle>Preview mode</AlertTitle>
                        <AlertDescription>
                          Showing the selected section and adjacent context. Sign in to
                          view the full agreement.
                        </AlertDescription>
                      </Alert>
                    </div>
                  ) : null}
                  <XMLRenderer
                    xmlContent={agreement.xml}
                    mode="agreement"
                    className={cn(
                      "text-sm leading-relaxed",
                      isMobile && "hyphens-auto",
                    )}
                    highlightedSection={highlightedSection}
                    isMobile={isMobile}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
