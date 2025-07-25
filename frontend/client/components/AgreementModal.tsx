import { useEffect, useState, useRef } from "react";
import {
  X,
  ArrowLeft,
  PanelLeftOpen,
  PanelLeftClose,
  ExternalLink,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAgreement } from "@/hooks/use-agreement";
import { XMLRenderer } from "./XMLRenderer";
import { TableOfContents } from "./TableOfContents";

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
  const [highlightedSection, setHighlightedSection] = useState<string | null>(
    null,
  );
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && agreementUuid) {
      fetchAgreement(agreementUuid);
    } else if (!isOpen) {
      clearAgreement();
    }
  }, [isOpen, agreementUuid, fetchAgreement, clearAgreement]);

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
        'button[class*="text-gray-400"]',
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
        // Both articles and sections should scroll to show their headers at the top
        const scrollOptions: ScrollIntoViewOptions = {
          behavior: "smooth", // Using smooth for fast but controlled scroll
          block: "start", // Always scroll to start to show the header
        };

        sectionElement.scrollIntoView(scrollOptions);

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
    <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full h-full max-w-7xl max-h-[95vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div className="flex items-center gap-4">
            <button
              onClick={onClose}
              className="flex items-center gap-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              <span className="text-sm">Back to Search</span>
            </button>

            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="flex items-center gap-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              {sidebarCollapsed ? (
                <PanelLeftOpen className="w-4 h-4" />
              ) : (
                <PanelLeftClose className="w-4 h-4" />
              )}
              <span className="text-sm">
                {sidebarCollapsed ? "Show" : "Hide"} Table of Contents
              </span>
            </button>
          </div>

          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Agreement Metadata */}
        {(agreementMetadata || agreement) && (
          <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
            <div className="grid grid-cols-4 gap-6 text-sm">
              <div>
                <span className="font-medium text-material-text-secondary">
                  Year:
                </span>
                <div className="text-material-text-primary">
                  {agreementMetadata?.year || agreement?.year}
                </div>
              </div>
              <div>
                <span className="font-medium text-material-text-secondary">
                  Target:
                </span>
                <div className="text-material-text-primary">
                  {agreementMetadata?.target || agreement?.target}
                </div>
              </div>
              <div>
                <span className="font-medium text-material-text-secondary">
                  Acquirer:
                </span>
                <div className="text-material-text-primary">
                  {agreementMetadata?.acquirer || agreement?.acquirer}
                </div>
              </div>

              {/* Original Filing Link */}
              <div className="flex justify-end">
                {agreement?.url && (
                  <a
                    href={agreement.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-material-blue hover:text-blue-700 hover:bg-blue-50 rounded-md transition-colors group"
                    title="View original SEC filing"
                  >
                    <ExternalLink className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                    <span>Original Filing</span>
                  </a>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar */}
          {!sidebarCollapsed && (
            <div className="w-80 border-r border-gray-200 bg-gray-50 flex-shrink-0">
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
          <div className="flex-1 overflow-hidden">
            {isLoading && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="w-8 h-8 border-4 border-material-blue border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                  <p className="text-gray-600">Loading agreement...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-red-600">
                  <p className="mb-2">Failed to load agreement</p>
                  <p className="text-sm text-gray-500">{error}</p>
                </div>
              </div>
            )}

            {agreement && (
              <div
                ref={contentRef}
                className="h-full overflow-y-auto p-6"
                style={{
                  scrollbarWidth: "thin",
                  scrollbarColor: "#e5e7eb #f9fafb",
                }}
              >
                <div className="max-w-4xl mx-auto">
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
