import { useId, useState } from "react";
import { Link } from "react-router-dom";
import { Info } from "lucide-react";

import { TaxonomyNodePanel } from "@/components/taxonomy/TaxonomyNodePanel";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { useIsMobile } from "@/hooks/use-mobile";
import type { TaxonomyKind } from "@/hooks/use-taxonomy";

interface ClauseTypeInfoProps {
  kind: TaxonomyKind;
  /** The selected clause type to describe. */
  standardId: string;
  /** Label of the selected clause type, for the button's accessible name. */
  label: string;
}

/**
 * Info button next to the clause-type picker: the selected node's taxonomy
 * panel, in a popover on desktop and a bottom sheet on mobile. Parent and
 * child links inside the panel browse in place.
 */
export function ClauseTypeInfo({ kind, standardId, label }: ClauseTypeInfoProps) {
  const isMobile = useIsMobile();
  const titleId = useId();
  const [open, setOpen] = useState(false);
  const [viewingId, setViewingId] = useState(standardId);

  const onOpenChange = (next: boolean) => {
    if (next) setViewingId(standardId);
    setOpen(next);
  };

  const panel = (
    <>
      <TaxonomyNodePanel
        kind={kind}
        standardId={viewingId}
        onSelectNode={setViewingId}
        titleAs="h3"
        titleId={titleId}
      />
      <Link
        to="/taxonomy"
        className="mt-4 inline-block text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
      >
        Open the Taxonomy page
      </Link>
    </>
  );

  const trigger = (
    <button
      type="button"
      className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      aria-label={`About ${label}`}
      title={`About ${label}`}
      onClick={isMobile ? () => onOpenChange(true) : undefined}
    >
      <Info className="h-4 w-4" aria-hidden="true" />
    </button>
  );

  if (isMobile) {
    return (
      <>
        {trigger}
        <Sheet open={open} onOpenChange={onOpenChange}>
          <SheetContent side="bottom" className="max-h-[85vh] overflow-y-auto">
            <SheetTitle className="sr-only">About {label}</SheetTitle>
            <SheetDescription className="sr-only">
              Where this clause type sits in the taxonomy, with an example.
            </SheetDescription>
            {panel}
          </SheetContent>
        </Sheet>
      </>
    );
  }

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <PopoverTrigger asChild>{trigger}</PopoverTrigger>
      <PopoverContent
        align="end"
        aria-labelledby={titleId}
        className="max-h-[min(36rem,var(--radix-popover-content-available-height))] w-[26rem] max-w-[calc(100vw-2rem)] overflow-y-auto p-4"
      >
        {panel}
      </PopoverContent>
    </Popover>
  );
}
