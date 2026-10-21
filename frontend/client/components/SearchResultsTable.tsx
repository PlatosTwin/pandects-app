import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ChevronDown, Copy } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useCopyToClipboard } from "@/components/ui/copy-button";
import { FlagAsInaccurateButton } from "@/components/FlagAsInaccurateButton";
import { StarButton } from "@/components/StarButton";
import type { ResultsDensity } from "@/components/ResultsToolbar";
import { useToast } from "@/hooks/use-toast";
import { formatCompactCurrencyValue, formatEnumValue } from "@/lib/format-utils";
import { cn } from "@/lib/utils";
import type { SearchResult } from "@shared/sections";
import { ClampedText, MetaLine, Parties, VerifiedBadge } from "@/pages/search/ResultBits";
import { ResultRow, RowActions } from "@/pages/search/ResultRow";
import {
  collapseTaxonomyPath,
  groupSectionResults,
  type SectionResultGroup,
  type SectionResultItem,
} from "@/pages/search/search-ui";

export interface SearchResultsTableProps {
  searchResults: SearchResult[];
  /** 1-based position of the first row (for telemetry and labels). */
  startPosition: number;
  selectedResults: Set<string>;
  onToggleResultSelection: (resultId: string) => void;
  clauseTypePathByStandardId: Record<string, readonly string[]>;
  getAgreementHref: (result: SearchResult) => string;
  onOpenAgreement: (result: SearchResult, position: number) => void;
  density?: ResultsDensity;
  /** Keyword matches to highlight in section text. */
  highlightPattern?: RegExp | null;
  signInHref: string;
  /** Clause types searched for; a matching type's path is shown first. */
  preferredTypeIds?: readonly string[];
}

function sectionTitleOf(result: SearchResult): string {
  return result.section_title?.trim() || result.article_title?.trim() || "Untitled section";
}

function CopyMenu({ result, href, text }: { result: SearchResult; href: string; text: string }) {
  const { copy } = useCopyToClipboard();
  const { toast } = useToast();
  const run = async (text: string, what: string) => {
    const ok = await copy(text);
    toast(ok ? { title: `${what} copied` } : { title: "Copy failed", variant: "destructive" });
  };
  const absolute = () =>
    typeof window === "undefined" ? href : new URL(href, window.location.origin).toString();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-9 w-9 p-0 text-muted-foreground hover:text-foreground sm:h-8 sm:w-8"
          aria-label={`Copy from ${sectionTitleOf(result)}`}
          title="Copy…"
        >
          <Copy className="h-4 w-4" aria-hidden="true" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[12rem]">
        {text ? (
          <DropdownMenuItem onSelect={() => void run(text, "Section text")}>Copy section text</DropdownMenuItem>
        ) : null}
        <DropdownMenuItem onSelect={() => void run(absolute(), "Link")}>Copy link</DropdownMenuItem>
        <DropdownMenuItem onSelect={() => void run(result.section_uuid, "Section UUID")}>
          Copy section UUID
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => void run(result.agreement_uuid, "Agreement UUID")}>
          Copy agreement UUID
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface SectionRowProps {
  item: SectionResultItem;
  /** Lead rows show deal context; follow-on rows from the same deal skip it. */
  variant: "lead" | "follow";
  props: SearchResultsTableProps;
}

function SectionRow({ item, variant, props }: SectionRowProps) {
  const { result, position, text, redacted, empty } = item;
  const { density = "comfy", highlightPattern, clauseTypePathByStandardId } = props;
  const href = props.getAgreementHref(result);
  const title = sectionTitleOf(result);
  const deal = result.metadata ?? {};
  const preferred = props.preferredTypeIds ?? [];
  const pathId =
    result.standard_id.find((id) => preferred.includes(id) && clauseTypePathByStandardId[id]) ??
    result.standard_id.find((id) => clauseTypePathByStandardId[id]);
  const path = collapseTaxonomyPath(pathId ? clauseTypePathByStandardId[pathId] : undefined);
  const extraTypes = result.standard_id.filter((id) => clauseTypePathByStandardId[id]).length - 1;
  const onOpen = () => props.onOpenAgreement(result, position);

  const meta =
    variant === "lead" ? (
      <MetaLine
        items={[
          result.year ? <span className="tabular-nums">{result.year}</span> : null,
          <Parties target={result.target} acquirer={result.acquirer} />,
          deal.transaction_price_total != null ? (
            <span className="tabular-nums">{formatCompactCurrencyValue(deal.transaction_price_total)}</span>
          ) : null,
          deal.deal_status || deal.deal_type ? (
            <span className="inline-flex gap-1">
              {deal.deal_status ? <Badge variant="muted">{formatEnumValue(deal.deal_status)}</Badge> : null}
              {deal.deal_type ? <Badge variant="muted">{formatEnumValue(deal.deal_type)}</Badge> : null}
            </span>
          ) : null,
          result.verified ? <VerifiedBadge /> : null,
        ]}
      />
    ) : (
      <span className="text-xs">Same deal · {result.article_title?.trim() || "Section"}</span>
    );

  return (
    <ResultRow
      density={density}
      selected={props.selectedResults.has(result.id)}
      onToggleSelected={() => props.onToggleResultSelection(result.id)}
      selectLabel={`Select ${title} (${result.target || "unknown target"})`}
      meta={meta}
      title={
        <h3 className={cn("text-base font-semibold leading-snug", density === "compact" && "text-sm")}>
          <Link
            to={href}
            onClick={onOpen}
            className="text-foreground hover:text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {title}
          </Link>
        </h3>
      }
      subtitle={
        path.length > 0 ? (
          <span title={path.join(" › ")}>
            {path.join(" › ")}
            {extraTypes > 0 ? ` +${extraTypes} more` : ""}
          </span>
        ) : null
      }
      actions={
        <RowActions
          star={
            <StarButton
              itemType="section"
              itemUuid={result.section_uuid}
              context={{
                agreement_uuid: result.agreement_uuid,
                target: result.target,
                acquirer: result.acquirer,
                article_title: result.article_title,
                section_title: result.section_title,
                standard_id: result.standard_id,
                xml: result.xml,
                position,
              }}
              className="shrink-0"
            />
          }
          flag={
            <FlagAsInaccurateButton
              source="search_result"
              agreement_uuid={result.agreement_uuid}
              section_uuid={result.section_uuid}
              className="shrink-0"
            />
          }
          extra={<CopyMenu result={result} href={href} text={redacted ? "" : text} />}
          openHref={href}
          openLabel={`Open ${title} in ${result.target || "the agreement"}`}
          onOpen={onOpen}
        />
      }
    >
      {redacted ? (
        <p className="text-sm text-muted-foreground">
          <Link to={props.signInHref} className="font-medium text-primary hover:underline">
            Sign in
          </Link>{" "}
          to read this section.
        </p>
      ) : empty ? (
        <p className="text-sm text-muted-foreground">No stored text.</p>
      ) : (
        <ClampedText
          text={text}
          pattern={highlightPattern}
          lines={variant === "lead" && density === "comfy" ? 6 : 3}
        />
      )}
    </ResultRow>
  );
}

function SectionGroup({ group, props }: { group: SectionResultGroup; props: SearchResultsTableProps }) {
  const [showMore, setShowMore] = useState(false);
  const [showEmpty, setShowEmpty] = useState(false);
  const first = group.lead?.result ?? group.empty[0]?.result;
  const moreCount = group.more.length;
  const emptyCount = group.empty.length;

  const toggles = (
    <div className="flex flex-wrap gap-x-4 gap-y-1 pl-1 text-sm">
      {moreCount > 0 ? (
        <button
          type="button"
          onClick={() => setShowMore((prev) => !prev)}
          aria-expanded={showMore}
          className="inline-flex items-center gap-1 font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <ChevronDown className={cn("h-4 w-4 transition-transform", !showMore && "-rotate-90")} aria-hidden="true" />
          {showMore
            ? `Hide ${moreCount} more ${moreCount === 1 ? "section" : "sections"}`
            : `+${moreCount} more ${moreCount === 1 ? "section" : "sections"} from this deal`}
        </button>
      ) : null}
      {emptyCount > 0 ? (
        <button
          type="button"
          onClick={() => setShowEmpty((prev) => !prev)}
          aria-expanded={showEmpty}
          className="text-muted-foreground hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {showEmpty
            ? `Hide ${emptyCount === 1 ? "empty section" : "empty sections"}`
            : `${emptyCount} empty ${emptyCount === 1 ? "section" : "sections"} hidden`}
        </button>
      ) : null}
    </div>
  );

  return (
    <li className="space-y-2">
      {group.lead ? (
        <SectionRow item={group.lead} variant="lead" props={props} />
      ) : first ? (
        <div className="flex flex-wrap items-center gap-x-1.5 rounded-lg border border-dashed border-border px-4 py-2.5 text-sm text-muted-foreground">
          <MetaLine
            items={[
              first.year ? <span className="tabular-nums">{first.year}</span> : null,
              <Parties target={first.target} acquirer={first.acquirer} />,
            ]}
          />
        </div>
      ) : null}
      {moreCount > 0 || emptyCount > 0 ? toggles : null}
      {showMore || showEmpty ? (
        <ul className="space-y-2 border-l-2 border-border pl-3 sm:ml-4">
          {showMore
            ? group.more.map((item) => (
                <li key={item.result.id}>
                  <SectionRow item={item} variant="follow" props={props} />
                </li>
              ))
            : null}
          {showEmpty
            ? group.empty.map((item) => (
                <li key={item.result.id}>
                  <SectionRow item={item} variant="follow" props={props} />
                </li>
              ))
            : null}
        </ul>
      ) : null}
    </li>
  );
}

/** Section results: one card per run of hits from the same deal. */
export function SearchResultsTable(props: SearchResultsTableProps) {
  const { searchResults, startPosition, density = "comfy" } = props;
  const groups = useMemo(
    () => groupSectionResults(searchResults, startPosition),
    [searchResults, startPosition],
  );
  return (
    <ul aria-label="Section results" className={density === "compact" ? "space-y-2" : "space-y-3"}>
      {groups.map((group) => (
        <SectionGroup key={group.key} group={group} props={props} />
      ))}
    </ul>
  );
}
