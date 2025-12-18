import { useEffect, useState, useRef } from "react";
import {
  X,
  ArrowLeft,
  PanelLeftOpen,
  PanelLeftClose,
  ExternalLink,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAgreement } from "@/hooks/use-agreement";
import { XMLRenderer } from "./XMLRenderer";
import { TableOfContents } from "./TableOfContents";
import { useIsMobile } from "@/hooks/use-mobile";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  animateScrollTop,
  getScrollTopForElementInContainer,
} from "@/lib/scroll";

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
  const isMobile = useIsMobile();
  const year = agreementMetadata?.year ?? agreement?.year;
  const target = agreementMetadata?.target ?? agreement?.target;
  const acquirer = agreementMetadata?.acquirer ?? agreement?.acquirer;
  const mobileMetadataSummary = (() => {
    const parts: string[] = [];
    if (year) parts.push(year);
    const counterparties =
      target && acquirer ? `${target} → ${acquirer}` : target ?? acquirer;
    if (counterparties) parts.push(counterparties);
    return parts.join(" · ");
  })();

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
        const contentContainer = sectionElement.querySelector(".ml-2");
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

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-[1px] flex items-center justify-center p-0 sm:p-4"
      onClick={onClose}
    >
      <div
        className={cn(
          "bg-card text-foreground shadow-xl w-full flex flex-col overflow-hidden",
          // Mobile: full-screen modal for a native-app feel
          "h-[100dvh] max-h-[100dvh] rounded-none",
          // Desktop: centered, rounded
          "sm:h-full sm:max-h-[95vh] sm:rounded-lg sm:max-w-7xl",
        )}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/70">
          <div className="flex items-center justify-between gap-3 px-3 py-3 sm:p-4">
            <div className="flex items-center gap-2 sm:gap-4 min-w-0">
              <Button
                onClick={onClose}
                variant="ghost"
                size="sm"
                className="gap-2 text-muted-foreground hover:text-foreground"
              >
                <ArrowLeft className="h-4 w-4" />
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
                    <PanelLeftOpen className="h-4 w-4" />
                  ) : (
                    <PanelLeftClose className="h-4 w-4" />
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
                      <PanelLeftOpen className="h-4 w-4" />
                      Contents
                    </Button>
                  </SheetTrigger>
                  <SheetContent side="left" className="w-[340px] p-0">
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

            <Button
              onClick={onClose}
              variant="ghost"
              size="icon"
              className="h-10 w-10 text-muted-foreground hover:text-foreground"
              aria-label="Close"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
        </div>

        {/* Agreement Metadata */}
        {(agreementMetadata || agreement) && (
          <div className="border-b border-border bg-muted/40 px-4 py-3 sm:px-6 sm:py-4">
            {isMobile ? (
              <details className="group">
                <summary className="flex items-center justify-between gap-3 cursor-pointer select-none">
                  <div className="min-w-0 flex-1 text-sm font-medium text-foreground truncate">
                    {mobileMetadataSummary}
                  </div>
                  <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
                </summary>

                <div className="mt-3 grid grid-cols-1 gap-2 text-sm">
                  {year && (
                    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                      <span className="font-medium text-muted-foreground">
                        Year:
                      </span>
                      <span className="text-foreground">{year}</span>
                    </div>
                  )}
                  {target && (
                    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                      <span className="font-medium text-muted-foreground">
                        Target:
                      </span>
                      <span className="text-foreground break-words">
                        {target}
                      </span>
                    </div>
                  )}
                  {acquirer && (
                    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                      <span className="font-medium text-muted-foreground">
                        Acquirer:
                      </span>
                      <span className="text-foreground break-words">
                        {acquirer}
                      </span>
                    </div>
                  )}

                  {agreement?.url && (
                    <a
                      href={agreement.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label="Original Filing (opens in a new tab)"
                      className="inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/10 group w-fit"
                      title="View original SEC filing"
                    >
                      <ExternalLink className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                      <span>Original Filing</span>
                    </a>
                  )}
                </div>
              </details>
            ) : (
              <div className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2 sm:gap-6 lg:grid-cols-4">
                <div>
                  <span className="font-medium text-muted-foreground">
                    Year:
                  </span>
                  <div className="text-foreground">{year}</div>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">
                    Target:
                  </span>
                  <div className="text-foreground break-words">{target}</div>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">
                    Acquirer:
                  </span>
                  <div className="text-foreground break-words">{acquirer}</div>
                </div>

                {/* Original Filing Link */}
                <div className="flex justify-start lg:justify-end">
                  {agreement?.url && (
                    <a
                      href={agreement.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label="Original Filing (opens in a new tab)"
                      className="inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/10 group"
                      title="View original SEC filing"
                    >
                      <ExternalLink className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                      <span>Original Filing</span>
                    </a>
                  )}
                </div>
              </div>
            )}
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
                <div className="text-center">
                  <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading agreement...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-red-600">
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
                <div className="max-w-4xl mx-auto">
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
                    className="text-sm leading-relaxed"
                    highlightedSection={highlightedSection}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
