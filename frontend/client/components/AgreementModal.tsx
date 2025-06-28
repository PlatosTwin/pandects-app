import { useEffect, useState, useRef } from "react";
import { X, ArrowLeft, PanelLeftOpen, PanelLeftClose } from "lucide-react";
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
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && agreementUuid) {
      fetchAgreement(agreementUuid);
    } else if (!isOpen) {
      clearAgreement();
    }
  }, [isOpen, agreementUuid, fetchAgreement, clearAgreement]);

  const scrollToSection = (sectionUuid: string) => {
    if (!contentRef.current) return;

    // Find element with data-section-uuid attribute
    const sectionElement = contentRef.current.querySelector(
      `[data-section-uuid="${sectionUuid}"]`,
    );
    if (sectionElement) {
      sectionElement.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  useEffect(() => {
    if (agreement && targetSectionUuid) {
      // Delay scroll to allow content to render
      const timer = setTimeout(() => {
        scrollToSection(targetSectionUuid);
      }, 500);
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
            <div className="grid grid-cols-3 gap-6 text-sm">
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
                  onSectionClick={scrollToSection}
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
