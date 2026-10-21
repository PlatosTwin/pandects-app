import { useEffect, useRef, useState } from "react";
import { Check, Link2, MoreHorizontal } from "lucide-react";
import { CopyButton, useCopyToClipboard } from "@/components/ui/copy-button";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { StarButton } from "@/components/StarButton";
import { FlagAsInaccurateButton } from "@/components/FlagAsInaccurateButton";
import { useAuth } from "@/hooks/use-auth";
import { useFavorites } from "@/contexts/FavoritesContext";
import { buildSectionHref } from "@/lib/agreement-links";
import { cn } from "@/lib/utils";

export interface SectionActionsProps {
  agreementUuid: string;
  sectionUuid: string;
  title: string;
  /** Builds the section's plain text on click. */
  getText: () => string;
  starContext: Record<string, unknown>;
  isMobile: boolean;
}

function sectionUrl(agreementUuid: string, sectionUuid: string) {
  return `${window.location.origin}${buildSectionHref(agreementUuid, sectionUuid)}`;
}

/** CopyButton with a link glyph (CopyButton always draws the copy glyph). */
function CopyLinkButton({ value, label }: { value: () => string; label: string }) {
  const { state, copy } = useCopyToClipboard();
  const name = state === "copied" ? "Link copied" : state === "failed" ? "Copy failed" : label;
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-8 w-8 text-muted-foreground hover:text-foreground [&_svg]:h-3.5 [&_svg]:w-3.5"
        aria-label={name}
        title={name}
        onClick={() => void copy(value())}
      >
        {state === "copied" ? (
          <Check className="text-success" aria-hidden="true" />
        ) : (
          <Link2 aria-hidden="true" />
        )}
      </Button>
      <span className="sr-only" role="status" aria-live="polite">
        {state === "copied" ? "Link copied" : state === "failed" ? "Copying failed" : ""}
      </span>
    </>
  );
}

function ActionButtons({
  agreementUuid,
  sectionUuid,
  title,
  getText,
  starContext,
  showStar,
  showFlag,
}: Omit<SectionActionsProps, "isMobile"> & { showStar: boolean; showFlag: boolean }) {
  return (
    <>
      <CopyLinkButton
        value={() => sectionUrl(agreementUuid, sectionUuid)}
        label={`Copy link to ${title}`}
      />
      <CopyButton
        value={getText}
        label={`Copy text of ${title}`}
        copiedMessage="Section text copied"
        className="[&_svg]:h-3.5 [&_svg]:w-3.5"
      />
      {showStar ? (
        <StarButton
          itemType="section"
          itemUuid={sectionUuid}
          context={starContext}
          buttonClassName="h-8 w-8"
          showTagSwatches={false}
        />
      ) : null}
      {showFlag ? (
        <FlagAsInaccurateButton
          // `search_result` is the source that carries a section_uuid; the
          // backend drops section context for `agreement_view` flags.
          source="search_result"
          agreement_uuid={agreementUuid}
          section_uuid={sectionUuid}
          label={`Report an issue with ${title}`}
          tooltipSide="top"
          className="h-8 w-8 sm:h-8 sm:w-8"
        />
      ) : null}
    </>
  );
}

/**
 * Hover/focus cluster on a section heading: copy link, copy text, star,
 * report issue. Star and report mount only once the heading is hovered or
 * the cluster takes focus, so a 500-section agreement doesn't mount 500
 * favorites lookups; a starred section keeps its filled star visible.
 */
export function SectionActions(props: SectionActionsProps) {
  const { sectionUuid, isMobile } = props;
  const { user } = useAuth();
  const { isStarred } = useFavorites();
  const starred = Boolean(user) && isStarred("section", sectionUuid);
  const [armed, setArmed] = useState(false);
  const rootRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (armed || isMobile) return;
    // The heading row (XMLRenderer's [data-section-heading]) is the hover target.
    const heading = rootRef.current?.closest("[data-section-heading]");
    if (!heading) return;
    const arm = () => setArmed(true);
    heading.addEventListener("pointerenter", arm);
    heading.addEventListener("focusin", arm);
    return () => {
      heading.removeEventListener("pointerenter", arm);
      heading.removeEventListener("focusin", arm);
    };
  }, [armed, isMobile]);

  if (isMobile) {
    return (
      <span ref={rootRef} className="flex shrink-0 items-center print:hidden">
        {starred ? (
          <StarButton
            itemType="section"
            itemUuid={sectionUuid}
            context={props.starContext}
            buttonClassName="h-8 w-8"
            showTagSwatches={false}
          />
        ) : null}
        <Popover>
          <PopoverTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-muted-foreground"
              aria-label={`Actions for ${props.title}`}
            >
              <MoreHorizontal aria-hidden="true" />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="end" className="flex w-auto items-center gap-0.5 p-1">
            <ActionButtons {...props} showStar={!starred} showFlag />
          </PopoverContent>
        </Popover>
      </span>
    );
  }

  return (
    <span
      ref={rootRef}
      data-section-actions=""
      className="-my-1 flex shrink-0 items-center print:hidden"
    >
      <span
        className={cn(
          "flex items-center opacity-0 transition-opacity focus-within:opacity-100 group-hover/heading:opacity-100",
          // Keep the cluster visible while one of its dialogs/popovers is open.
          "has-[[data-state=open]]:opacity-100",
        )}
      >
        <ActionButtons {...props} showStar={armed && !starred} showFlag={armed} />
      </span>
      {starred ? (
        <StarButton
          itemType="section"
          itemUuid={sectionUuid}
          context={props.starContext}
          buttonClassName="h-8 w-8"
          showTagSwatches={false}
        />
      ) : null}
    </span>
  );
}
