import { Check, Copy, Download } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useCopyToClipboard } from "@/components/ui/copy-button";

export interface TaxonomyExportButtonsProps {
  /** Built lazily so the tree text is only rendered when a button is used. */
  getText: () => string;
  fileName: string;
  disabled?: boolean;
}

/**
 * Copy-to-clipboard and download-as-.txt controls for the taxonomy tree.
 * Both emit the same tree-shaped plain text regardless of the view mode
 * currently selected on the page.
 */
export function TaxonomyExportButtons({
  getText,
  fileName,
  disabled = false,
}: TaxonomyExportButtonsProps) {
  const { state: copyState, copy } = useCopyToClipboard();

  const handleCopy = () => copy(getText());

  const handleDownload = () => {
    const blob = new Blob([getText()], { type: "text/plain;charset=utf-8;" });
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
  };

  const copyLabel =
    copyState === "copied"
      ? "Copied"
      : copyState === "failed"
        ? "Copy failed"
        : "Copy";

  return (
    <div className="flex items-center gap-2">
      <Button
        type="button"
        variant="outline"
        onClick={() => void handleCopy()}
        disabled={disabled}
        className="h-10 px-3"
        // Folds the transient state into the name so it never contradicts the
        // visible label (WCAG 2.5.3).
        aria-label={
          copyState === "idle" ? "Copy taxonomy tree as text" : copyLabel
        }
      >
        {copyState === "copied" ? (
          <Check className="h-4 w-4" aria-hidden="true" />
        ) : (
          <Copy className="h-4 w-4" aria-hidden="true" />
        )}
        {copyLabel}
      </Button>
      <Button
        type="button"
        variant="outline"
        onClick={handleDownload}
        disabled={disabled}
        className="h-10 px-3"
        aria-label="Download taxonomy tree as a text file"
      >
        <Download className="h-4 w-4" aria-hidden="true" />
        Download
      </Button>
      <span className="sr-only" role="status" aria-live="polite">
        {copyState === "copied"
          ? "Taxonomy tree copied to clipboard"
          : copyState === "failed"
            ? "Copying the taxonomy tree failed"
            : ""}
      </span>
    </div>
  );
}
