import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useTransition,
} from "react";
import type { ReactNode } from "react";
import { Link, useLocation, useNavigate, useNavigationType } from "react-router-dom";
import {
  ArrowLeft,
  FileSearch,
  Info,
  ListTree,
  MoreHorizontal,
  Search,
  X,
} from "lucide-react";
import { StarButton } from "@/components/StarButton";
import { useAgreementQuery, useAgreementSections, useReaderIndex } from "@/hooks/use-agreement";
import { useAuth } from "@/hooks/use-auth";
import { useIsMobile } from "@/hooks/use-mobile";
import { useScrollSpy } from "@/hooks/use-scroll-spy";
import { useToast } from "@/hooks/use-toast";
import { useFavorites } from "@/contexts/FavoritesContext";
import { XMLRenderer } from "@/components/XMLRenderer";
import { ErrorState } from "@/components/StateMessage";
import { Button } from "@/components/ui/button";
import { ExternalLink } from "@/components/ui/external-link";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { useCopyToClipboard } from "@/components/ui/copy-button";
import { FindBar } from "@/components/reader/FindBar";
import { MatchStrip } from "@/components/reader/MatchStrip";
import { ReaderBackLink } from "@/components/reader/ReaderBackLink";
import { ReaderContents, buildSectionTypeOptions } from "@/components/reader/ReaderContents";
import { useTaxonomy } from "@/hooks/use-taxonomy";
import { indexClauseTypePaths } from "@/lib/clause-type-index";
import { ReaderDetails } from "@/components/reader/ReaderDetails";
import { SectionActions } from "@/components/reader/SectionActions";
import {
  isSmartReaderEnabled,
  readContentsOpen,
  readTermHighlightMode,
  writeContentsOpen,
  writeTermHighlightMode,
  type TermHighlightMode,
} from "@/components/reader/reader-preferences";
import { ReaderOverlayLayer, outlineNodeName } from "@/components/reader/ReaderOverlayLayer";
import {
  anchorKey,
  outlineTrail,
  type ReaderOutlineNode,
  type ReaderOverlay,
  type ReaderXref,
} from "@/lib/reader-index";
import { extractSectionPlainText, sectionShortLabel } from "@/components/reader/section-text";
import { useFindInDocument } from "@/components/reader/use-find-in-document";
import { useMediaQuery } from "@/components/reader/use-media-query";
import { useReaderMatches } from "@/components/reader/use-reader-data";
import {
  buildAgreementHref,
  parseAgreementOrigin,
} from "@/lib/agreement-links";
import { buildAccountPathWithNext } from "@/lib/auth-next";
import {
  formatBooleanValue,
  formatCompactCurrencyValue,
  formatDateValue,
  formatEnumValue,
  formatTextValue,
} from "@/lib/format-utils";
import { formatIndustry } from "@/lib/naics";
import { prefersReducedMotion } from "@/lib/scroll";
import { decodeXmlEntities } from "@/lib/text-utils";
import { cn } from "@/lib/utils";

interface AgreementReaderProps {
  agreementUuid: string;
  focusSectionUuid?: string | null;
  /** The `from` query param: app path the reader was opened from (see lib/agreement-links). */
  backTo?: string | null;
}

type AgreementTextSize = "small" | "medium" | "large";

/** Site navigation bar height (Navigation.tsx `h-16`); the reader header pins below it. */
const SITE_NAV_HEIGHT = 64;
const DESKTOP_CONTENTS_QUERY = "(min-width: 1024px)";

interface DocSectionInfo {
  title: string;
  order: number;
}

const HEADING_TAG_RE = /<(article|section)\b([^>]*)>/g;
const UUID_ATTR_RE = /\buuid="([^"]*)"/;
const TITLE_ATTR_RE = /\btitle="([^"]*)"/;

/** uuid -> title + document order for every article/section in the XML. */
function indexDocumentSections(xml: string): Map<string, DocSectionInfo> {
  const map = new Map<string, DocSectionInfo>();
  let order = 0;
  for (const match of xml.matchAll(HEADING_TAG_RE)) {
    const uuid = match[2].match(UUID_ATTR_RE)?.[1];
    if (!uuid || map.has(uuid)) continue;
    map.set(uuid, { title: match[2].match(TITLE_ATTR_RE)?.[1] ?? "", order: order++ });
  }
  return map;
}

/** Containers that are regions, not sections; the scroll-spy ids them by element id. */
const REGION_CONTAINERS = new Set(["frontMatter", "tableOfContents", "body", "sigPages", "backMatter"]);
const FLASH_MS = 1800;

/** History state the smart reader puts on entries it creates or leaves. */
interface ReaderHistoryState {
  /** Scroll position to restore when the user comes Back to this entry. */
  readerY?: number;
  /** Set on the entry a cross-reference pushed: the outline id it went to. */
  readerJump?: string;
  /** "Section 5.2(c)": where that cross-reference was. */
  readerBack?: string;
}

function spyIdFor(container: string): string {
  return REGION_CONTAINERS.has(container) ? `agreement-region-${container}` : container;
}

/** The section, or subsection, an anchor key sits in, for the Back chip. */
function placeOfAnchor(overlay: ReaderOverlay, key: string): string | null {
  const cut = key.lastIndexOf("/");
  const container = key.slice(0, cut);
  const t = Number(key.slice(cut + 1));
  const node = overlay.outlineByUuid.get(container);
  if (!node) return null;
  let place: ReaderOutlineNode = node;
  for (const child of node.children ?? []) {
    if (child.anchor.s === container && child.anchor.t != null && child.anchor.t <= t) place = child;
  }
  return outlineNodeName(place);
}

function errorFromMessage(message: string | null) {
  const status = message?.match(/HTTP (\d{3})/)?.[1];
  return { status: status ? Number(status) : undefined, message: message ?? undefined };
}

function isMacPlatform() {
  if (typeof navigator === "undefined") return false;
  return /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
}

function ReaderSkeleton() {
  return (
    <div className="app-frame py-6" role="status">
      <span className="sr-only">Loading agreement</span>
      <div aria-hidden="true" className="space-y-6">
        <div className="flex h-12 items-center gap-3">
          <Skeleton className="h-6 w-24" />
          <Skeleton className="h-6 flex-1" />
        </div>
        <div className="grid gap-8 lg:grid-cols-[300px_minmax(0,1fr)]">
          <Skeleton className="hidden h-96 w-full lg:block" />
          <div className="mx-auto w-full max-w-[calc(72ch+4rem)] space-y-3">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-[60vh] w-full" />
          </div>
        </div>
      </div>
    </div>
  );
}

export function AgreementReader({
  agreementUuid,
  focusSectionUuid,
  backTo,
}: AgreementReaderProps) {
  // With a focus section the API redacts around it (focus + neighbors), so
  // each (uuid, focus) pair is its own cached query.
  const {
    agreement,
    isLoading,
    error,
    query: agreementQuery,
  } = useAgreementQuery(agreementUuid, focusSectionUuid);
  const { sections: sectionIndex, error: sectionIndexError } =
    useAgreementSections(agreementUuid);
  // Parent labels for the section-type chips (the index only has leaf labels).
  const { taxonomyTree } = useTaxonomy({ enabled: sectionIndex.length > 0 });

  const origin = useMemo(() => parseAgreementOrigin(backTo), [backTo]);
  const matchesQuery = useReaderMatches(origin, agreementUuid);
  const isMobile = useIsMobile();
  const isDesktopContents = useMediaQuery(DESKTOP_CONTENTS_QUERY);
  const { user } = useAuth();
  const { ensureLoaded } = useFavorites();
  const { toast } = useToast();
  const { copy } = useCopyToClipboard();

  const [contentsOpen, setContentsOpenState] = useState(readContentsOpen);
  const [tocSheetOpen, setTocSheetOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [detailsSheetOpen, setDetailsSheetOpen] = useState(false);
  const [showBodyOnly, setShowBodyOnly] = useState(false);
  const [textSize, setTextSize] = useState<AgreementTextSize>("medium");
  const [sectionType, setSectionType] = useState("all");
  const [findOpen, setFindOpen] = useState(false);
  const [findQuery, setFindQuery] = useState("");
  const [currentMatchUuid, setCurrentMatchUuid] = useState<string | null>(
    focusSectionUuid ?? null,
  );
  const [matchStripOpen, setMatchStripOpen] = useState(true);
  const [jumpingUuid, setJumpingUuid] = useState<string | null>(focusSectionUuid ?? null);
  const [revealRequest, setRevealRequest] = useState<{ uuid: string; seq: number } | null>(
    null,
  );
  const [stickyBottom, setStickyBottom] = useState(SITE_NAV_HEIGHT + 48);
  // ---- Smart reader (flagged) ---------------------------------------------
  const [smartReader] = useState(isSmartReaderEnabled);
  const [termMode, setTermModeState] = useState<TermHighlightMode>(readTermHighlightMode);
  const [overlay, setOverlay] = useState<ReaderOverlay | null>(null);
  const readerIndex = useReaderIndex(agreementUuid, focusSectionUuid, smartReader);
  const navigate = useNavigate();
  const location = useLocation();
  const navigationType = useNavigationType();
  const historyState = (location.state ?? null) as ReaderHistoryState | null;
  // The document renders in a transition so a 500-section agreement doesn't
  // freeze the header while it mounts; the skeleton stays up meanwhile.
  const [renderedXml, setRenderedXml] = useState<string | null>(null);
  const [, startDocTransition] = useTransition();

  const headerRef = useRef<HTMLElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const findInputRef = useRef<HTMLInputElement>(null);
  const findButtonRef = useRef<HTMLButtonElement>(null);
  const stickyBottomRef = useRef(stickyBottom);
  const revealSeqRef = useRef(0);
  const jumpTimerRef = useRef<number | null>(null);
  const flashTimerRef = useRef<number | null>(null);
  const handledLocationKeyRef = useRef<string | null>(null);

  const xml = agreement?.xml ?? null;
  const docReady = xml !== null && renderedXml === xml;

  useEffect(() => {
    if (xml === null) return;
    startDocTransition(() => setRenderedXml(xml));
  }, [xml]);

  const setTermMode = useCallback((mode: TermHighlightMode) => {
    setTermModeState(mode);
    writeTermHighlightMode(mode);
  }, []);

  const setContentsOpen = useCallback((open: boolean) => {
    setContentsOpenState(open);
    writeContentsOpen(open);
  }, []);

  // ---- Sticky header measurement -------------------------------------------
  useEffect(() => {
    const element = headerRef.current;
    if (!element) return;
    const update = () => {
      // `top-16` is 4rem, which tracks the root font size.
      const stickyTop = parseFloat(window.getComputedStyle(element).top) || SITE_NAV_HEIGHT;
      const next = stickyTop + element.getBoundingClientRect().height;
      stickyBottomRef.current = next;
      setStickyBottom(next);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, [agreement]);

  const { activeId: spyActiveId, notifyJump } = useScrollSpy({
    containerRef: contentRef,
    topOffset: stickyBottom,
    contentKey: docReady ? renderedXml : null,
    enabled: docReady,
  });

  // ---- Document indexes ----------------------------------------------------
  const docSections = useMemo(
    () => (xml ? indexDocumentSections(xml) : new Map<string, DocSectionInfo>()),
    [xml],
  );

  const clauseTypePathById = useMemo(
    () => (taxonomyTree ? indexClauseTypePaths(taxonomyTree) : {}),
    [taxonomyTree],
  );
  const sectionTypeOptions = useMemo(
    () => buildSectionTypeOptions(sectionIndex, clauseTypePathById),
    [sectionIndex, clauseTypePathById],
  );

  const visibleSectionUuids = useMemo(() => {
    if (sectionType === "all") return null;
    return new Set(
      sectionIndex
        .filter((item) => item.standard_id.includes(sectionType))
        .map((item) => item.section_uuid),
    );
  }, [sectionIndex, sectionType]);

  // Batch-load section favorites once so starred sections show their star
  // without each heading asking separately.
  useEffect(() => {
    if (!user || sectionIndex.length === 0) return;
    ensureLoaded(
      "section",
      sectionIndex.map((item) => item.section_uuid),
    );
  }, [ensureLoaded, sectionIndex, user]);

  // ---- Matches from the originating search ---------------------------------
  const matchedUuids = useMemo(() => {
    const fromSearch = (matchesQuery.data?.sectionUuids ?? []).filter((uuid) =>
      docSections.has(uuid),
    );
    const list = new Set(fromSearch);
    if (focusSectionUuid && docSections.has(focusSectionUuid)) list.add(focusSectionUuid);
    return Array.from(list).sort(
      (a, b) => (docSections.get(a)?.order ?? 0) - (docSections.get(b)?.order ?? 0),
    );
  }, [docSections, focusSectionUuid, matchesQuery.data]);
  const matchedSet = useMemo(() => new Set(matchedUuids), [matchedUuids]);
  const matchTotal = Math.max(matchedUuids.length, matchesQuery.data?.totalCount ?? 0);
  const matchHasMore = matchTotal > matchedUuids.length;
  const currentMatchIndex = currentMatchUuid ? matchedUuids.indexOf(currentMatchUuid) : -1;

  // ---- Jumping ------------------------------------------------------------
  const finishJump = useCallback(() => {
    if (jumpTimerRef.current !== null) window.clearTimeout(jumpTimerRef.current);
    jumpTimerRef.current = null;
    setJumpingUuid(null);
  }, []);

  const scrollToElement = useCallback(
    (element: Element, instant: boolean) => {
      const rect = element.getBoundingClientRect();
      const top = Math.max(0, window.scrollY + rect.top - stickyBottomRef.current - 16);
      const distance = Math.abs(top - window.scrollY);
      // Smooth-scrolling across a long agreement takes seconds; jump instead.
      const smooth =
        !instant && !prefersReducedMotion() && distance < window.innerHeight * 3;
      window.scrollTo({ top, behavior: smooth ? "smooth" : "auto" });
      if (jumpTimerRef.current !== null) window.clearTimeout(jumpTimerRef.current);
      jumpTimerRef.current = window.setTimeout(finishJump, smooth ? 700 : 150);
    },
    [finishJump],
  );

  const flashElement = useCallback((element: Element) => {
    if (flashTimerRef.current !== null) window.clearTimeout(flashTimerRef.current);
    element.classList.remove("reader-flash");
    // Restart the animation when the same target is flashed twice.
    void (element as HTMLElement).offsetWidth;
    element.classList.add("reader-flash");
    flashTimerRef.current = window.setTimeout(() => element.classList.remove("reader-flash"), FLASH_MS);
  }, []);

  /**
   * Expand the collapsed ancestors of `container` (a section/article uuid or
   * a region), then scroll to the element `selector` finds.
   */
  const scrollToTarget = useCallback(
    (
      selector: string,
      container: string,
      options?: { instant?: boolean; flash?: boolean },
    ) => {
      revealSeqRef.current += 1;
      setRevealRequest({ uuid: container, seq: revealSeqRef.current });
      setJumpingUuid(container);
      notifyJump(spyIdFor(container));
      setTocSheetOpen(false);
      const attempt = (retriesLeft: number) => {
        const element = contentRef.current?.querySelector(selector);
        if (!element) {
          // The target may sit inside a collapsed article: the reveal request
          // expands it on the next render, so retry briefly.
          if (retriesLeft > 0) window.setTimeout(() => attempt(retriesLeft - 1), 80);
          else finishJump();
          return;
        }
        scrollToElement(element, options?.instant ?? false);
        if (options?.flash) flashElement(element);
      };
      attempt(6);
    },
    [finishJump, flashElement, notifyJump, scrollToElement],
  );

  const scrollToSection = useCallback(
    (sectionUuid: string, options?: { instant?: boolean }) =>
      scrollToTarget(`[data-section-uuid="${sectionUuid}"]`, sectionUuid, options),
    [scrollToTarget],
  );

  const scrollToTextAnchor = useCallback(
    (key: string, container: string, options?: { instant?: boolean }) =>
      scrollToTarget(`[data-anchor="${CSS.escape(key)}"]`, container, { ...options, flash: true }),
    [scrollToTarget],
  );

  /** Jump to an outline node: sections and articles by uuid, subsections by text anchor. */
  const jumpToOutline = useCallback(
    (id: string, options?: { instant?: boolean }): boolean => {
      const node = overlay?.outlineById.get(id);
      if (!node) return false;
      if (node.anchor.t == null) scrollToSection(node.anchor.s, options);
      else scrollToTextAnchor(anchorKey(node.anchor), node.anchor.s, options);
      return true;
    },
    [overlay, scrollToSection, scrollToTextAnchor],
  );

  // A cross-reference jump is a history entry: Back returns to the citing
  // text (the entry left behind keeps its scroll position), and the new
  // entry's hash is the target's outline id, so the URL can be shared.
  const followXref = useCallback(
    (xref: ReaderXref, source: HTMLElement) => {
      if (!xref.target || !overlay?.outlineById.has(xref.target)) return;
      const sourceKey = source.closest("[data-anchor]")?.getAttribute("data-anchor");
      const here = { pathname: window.location.pathname, search: window.location.search };
      navigate(
        { ...here, hash: window.location.hash },
        { replace: true, state: { ...historyState, readerY: window.scrollY } satisfies ReaderHistoryState },
      );
      navigate(
        { ...here, hash: `#${encodeURIComponent(xref.target)}` },
        {
          state: {
            readerJump: xref.target,
            readerBack: (sourceKey && placeOfAnchor(overlay, sourceKey)) || undefined,
          } satisfies ReaderHistoryState,
        },
      );
      jumpToOutline(xref.target);
    },
    [historyState, jumpToOutline, navigate, overlay],
  );

  const scrollToAnchor = useCallback(
    (anchorId: string) => {
      const element = document.getElementById(anchorId);
      if (!element) return;
      notifyJump(anchorId);
      setTocSheetOpen(false);
      scrollToElement(element, false);
    },
    [notifyJump, scrollToElement],
  );

  useEffect(
    () => () => {
      if (jumpTimerRef.current !== null) window.clearTimeout(jumpTimerRef.current);
      if (flashTimerRef.current !== null) window.clearTimeout(flashTimerRef.current);
    },
    [],
  );

  // Back/Forward and deep links. Entries this reader pushed are handled where
  // they're pushed; a POP restores the scroll position the entry was left at,
  // or resolves its `#s4.3(b)` hash through the outline. On first load a
  // focus section (from search) wins over a hash.
  useEffect(() => {
    if (!overlay || !docReady) return;
    if (handledLocationKeyRef.current === location.key) return;
    const isFirst = handledLocationKeyRef.current === null;
    handledLocationKeyRef.current = location.key;
    if (navigationType !== "POP") return;
    if (historyState?.readerY != null) {
      window.scrollTo({ top: historyState.readerY });
      return;
    }
    if (isFirst && focusSectionUuid) return;
    const hash = location.hash.slice(1);
    if (!hash) return;
    let id = hash;
    try {
      id = decodeURIComponent(hash);
    } catch {
      // A malformed escape: try the raw hash.
    }
    jumpToOutline(id, { instant: isFirst });
  }, [
    docReady,
    focusSectionUuid,
    historyState,
    jumpToOutline,
    location.hash,
    location.key,
    navigationType,
    overlay,
  ]);

  // A new focus (e.g. a pasted section link within the same agreement)
  // becomes the current match.
  const [previousFocus, setPreviousFocus] = useState(focusSectionUuid);
  if (previousFocus !== focusSectionUuid) {
    setPreviousFocus(focusSectionUuid);
    setCurrentMatchUuid(focusSectionUuid ?? null);
    setJumpingUuid(focusSectionUuid ?? null);
    setMatchStripOpen(true);
  }

  // Deep link / arrival from search: jump once the document has rendered.
  useEffect(() => {
    if (!focusSectionUuid || !docReady) return;
    const frame = window.requestAnimationFrame(() =>
      scrollToSection(focusSectionUuid, { instant: true }),
    );
    return () => window.cancelAnimationFrame(frame);
  }, [docReady, focusSectionUuid, scrollToSection]);

  const goToMatch = useCallback(
    (index: number) => {
      if (matchedUuids.length === 0) return;
      const wrapped = (index + matchedUuids.length) % matchedUuids.length;
      const uuid = matchedUuids[wrapped];
      setCurrentMatchUuid(uuid);
      scrollToSection(uuid);
    },
    [matchedUuids, scrollToSection],
  );

  // ---- Find in agreement ----------------------------------------------------
  const find = useFindInDocument({
    containerRef: contentRef,
    query: findOpen ? findQuery : "",
    topOffset: stickyBottom,
    contentKey: docReady ? renderedXml : null,
  });

  const openFind = useCallback(() => {
    setFindOpen(true);
    window.requestAnimationFrame(() => {
      findInputRef.current?.focus();
      findInputRef.current?.select();
    });
  }, []);

  const closeFind = useCallback(() => {
    setFindOpen(false);
    setFindQuery("");
    window.requestAnimationFrame(() => findButtonRef.current?.focus());
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey) return;
      if (event.key.toLowerCase() !== "f") return;
      // A second ⌘F while the find input has focus falls through to the
      // browser's own find.
      if (document.activeElement === findInputRef.current) return;
      event.preventDefault();
      openFind();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [openFind]);

  // ---- Section actions ------------------------------------------------------
  const target = agreement?.target ?? null;
  const acquirer = agreement?.acquirer ?? null;
  const renderSectionActions = useCallback(
    (section: { uuid: string; title: string }): ReactNode => (
      <SectionActions
        agreementUuid={agreementUuid}
        sectionUuid={section.uuid}
        title={section.title}
        getText={() => (xml ? extractSectionPlainText(xml, section.uuid) : null) ?? section.title}
        starContext={{
          agreement_uuid: agreementUuid,
          target,
          acquirer,
          section_title: section.title,
        }}
        isMobile={isMobile}
      />
    ),
    [acquirer, agreementUuid, isMobile, target, xml],
  );

  // ---- Panels ----------------------------------------------------------------
  const showContentsPanel = isDesktopContents && contentsOpen;

  // Breadcrumb for the section the scroll-spy says is current.
  const breadcrumb = useMemo(() => {
    const node = spyActiveId ? overlay?.outlineByUuid.get(spyActiveId) : undefined;
    if (!overlay || !node) return null;
    return outlineTrail(overlay, node.id).map((item) => ({
      id: item.id,
      text:
        item.kind === "article"
          ? outlineNodeName(item)
          : decodeXmlEntities(item.title) || outlineNodeName(item),
    }));
  }, [overlay, spyActiveId]);

  const subsectionsFor = useCallback(
    (sectionUuid: string) =>
      (overlay?.outlineByUuid.get(sectionUuid)?.children ?? []).filter(
        (child) => child.kind === "sub",
      ),
    [overlay],
  );
  const toggleContents = () => {
    if (isDesktopContents) setContentsOpen(!contentsOpen);
    else setTocSheetOpen(true);
  };

  // ---- Render ----------------------------------------------------------------
  if (isLoading && !agreement) {
    return <ReaderSkeleton />;
  }

  if (error || !agreement) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-10 sm:px-6 lg:px-8">
        <ErrorState
          // The query's own error keeps its status, so a missing agreement
          // reads as "not found" and an outage as retryable.
          error={agreementQuery.error ?? errorFromMessage(error)}
          title="Agreement unavailable"
          onRetry={() => void agreementQuery.refetch()}
          action={
            <Button asChild variant="outline" size="sm">
              <Link to={origin?.href ?? "/search"}>
                {origin?.label ?? "Go to Search"}
              </Link>
            </Button>
          }
        />
      </div>
    );
  }

  const targetName = agreement.target?.trim() || "";
  const acquirerName = agreement.acquirer?.trim() || "";
  const title = targetName || acquirerName || "Agreement";
  const showAcquirer = Boolean(targetName && acquirerName);

  const targetIndustryDisplay = formatIndustry(
    agreement.target_industry,
    agreement.target_industry_label,
  );
  const acquirerIndustryDisplay = formatIndustry(
    agreement.acquirer_industry,
    agreement.acquirer_industry_label,
  );
  const metadata = [
    { label: "Target", value: formatTextValue(agreement.target) },
    { label: "Acquirer", value: formatTextValue(agreement.acquirer) },
    { label: "Filed", value: formatDateValue(agreement.filing_date) },
    { label: "Announced", value: formatDateValue(agreement.announce_date) },
    { label: "Closed", value: formatDateValue(agreement.close_date) },
    { label: "Deal type", value: formatEnumValue(agreement.deal_type) },
    { label: "Status", value: formatEnumValue(agreement.deal_status) },
    { label: "Attitude", value: formatEnumValue(agreement.attitude) },
    { label: "Purpose", value: formatEnumValue(agreement.purpose) },
    { label: "Consideration", value: formatEnumValue(agreement.transaction_consideration) },
    { label: "Value", value: formatCompactCurrencyValue(agreement.transaction_price_total) },
    { label: "Target industry", value: formatTextValue(targetIndustryDisplay) },
    { label: "Acquirer industry", value: formatTextValue(acquirerIndustryDisplay) },
    { label: "Target PE", value: formatBooleanValue(agreement.target_pe) },
    { label: "Acquirer PE", value: formatBooleanValue(agreement.acquirer_pe) },
  ];

  const facts: Array<{ key: string; label: string; value: string }> = [];
  if (agreement.year) facts.push({ key: "year", label: "Year", value: String(agreement.year) });
  if (agreement.transaction_price_total !== null) {
    facts.push({
      key: "value",
      label: "Value",
      value: formatCompactCurrencyValue(agreement.transaction_price_total),
    });
  }
  if (agreement.deal_type) {
    facts.push({ key: "type", label: "Deal type", value: formatEnumValue(agreement.deal_type) });
  }
  if (agreement.deal_status) {
    facts.push({ key: "status", label: "Status", value: formatEnumValue(agreement.deal_status) });
  }
  if (agreement.filing_date) {
    facts.push({
      key: "filed",
      label: "Filed",
      value: `Filed ${formatDateValue(agreement.filing_date)}`,
    });
  }

  const agreementTextSizeClass =
    textSize === "small"
      ? "text-sm"
      : textSize === "large"
        ? "text-base sm:text-lg"
        : "text-sm sm:text-base";

  const currentMatchTitle = currentMatchUuid
    ? docSections.get(currentMatchUuid)?.title
    : undefined;
  const showMatchStrip = matchStripOpen && matchedUuids.length > 0;
  const matchStrip = showMatchStrip ? (
    <MatchStrip
      label={
        currentMatchUuid
          ? sectionShortLabel(currentMatchTitle)
          : `${matchTotal}${matchHasMore ? "+" : ""} sections`
      }
      index={currentMatchIndex}
      total={matchedUuids.length}
      hasMore={matchHasMore}
      onPrevious={() => goToMatch(currentMatchIndex <= 0 ? -1 : currentMatchIndex - 1)}
      onNext={() => goToMatch(currentMatchIndex + 1)}
      onDismiss={() => setMatchStripOpen(false)}
      backHref={
        origin && (origin.kind === "search" || origin.kind === "comparison")
          ? origin.href
          : null
      }
      jumping={jumpingUuid !== null && jumpingUuid === currentMatchUuid}
    />
  ) : null;

  const contents = (
    <ReaderContents
      xmlContent={agreement.xml}
      activeItemId={spyActiveId}
      targetSectionUuid={currentMatchUuid ?? undefined}
      onSectionClick={(uuid) => scrollToSection(uuid)}
      onAnchorClick={scrollToAnchor}
      sectionTypeOptions={sectionTypeOptions}
      sectionType={sectionType}
      onSectionTypeChange={setSectionType}
      visibleSectionUuids={visibleSectionUuids}
      matchedSectionUuids={matchedSet}
      sectionIndexError={sectionIndexError}
      subsectionsFor={overlay ? subsectionsFor : undefined}
      onSubsectionClick={(id) => jumpToOutline(id)}
    />
  );

  const details = <ReaderDetails metadata={metadata} filingUrl={agreement.url ?? null} />;
  const shortcutHint = isMacPlatform() ? "⌘F" : "Ctrl F";
  const iconButtonClass = "h-10 w-10 shrink-0 text-muted-foreground hover:text-foreground";
  const labeledButtonClass =
    "h-10 shrink-0 gap-1.5 px-2.5 text-muted-foreground hover:text-foreground";

  const findBar = (
    <FindBar
      ref={findInputRef}
      query={findQuery}
      onQueryChange={setFindQuery}
      onClose={closeFind}
      find={find}
      className={isMobile ? "flex-1" : "w-[min(24rem,38vw)]"}
    />
  );

  return (
    <div style={{ overflowX: "clip" }}>
      {/* The one sticky row. Opaque so body text never shows through. */}
      <header
        ref={headerRef}
        className="sticky top-16 z-30 border-b border-border bg-background print:hidden"
      >
        <div className="app-frame">
          <div className="relative flex h-12 items-center gap-1 sm:gap-2">
            <ReaderBackLink origin={origin} compact={isMobile} />
            <div className="flex min-w-0 flex-1 items-baseline gap-1.5">
              <h1
                id="agreement-reader-title"
                className="min-w-0 truncate text-base font-semibold text-foreground sm:text-lg"
                title={showAcquirer ? `${title}, acquired by ${acquirerName}` : title}
              >
                {title}
              </h1>
              {showAcquirer ? (
                <span
                  className="hidden min-w-0 shrink truncate text-sm text-muted-foreground lg:inline"
                  title={`Acquired by ${acquirerName}`}
                >
                  <span aria-hidden="true">← </span>
                  <span className="sr-only">acquired by </span>
                  {acquirerName}
                </span>
              ) : null}
            </div>

            {findOpen && !isMobile ? (
              findBar
            ) : (
              <Button
                ref={findButtonRef}
                type="button"
                variant="ghost"
                size="sm"
                className={isMobile ? iconButtonClass : labeledButtonClass}
                onClick={openFind}
                aria-label={isMobile ? "Find in agreement" : undefined}
                aria-keyshortcuts={isMacPlatform() ? "Meta+F" : "Control+F"}
              >
                <Search className="h-4 w-4" aria-hidden="true" />
                {isMobile ? null : (
                  <>
                    Find
                    <kbd className="ml-0.5 hidden rounded border border-border bg-muted px-1 font-sans text-[10px] font-medium text-muted-foreground md:inline">
                      {shortcutHint}
                    </kbd>
                  </>
                )}
              </Button>
            )}

            <Button
              type="button"
              variant="ghost"
              size="sm"
              className={cn(
                isMobile ? iconButtonClass : labeledButtonClass,
                showContentsPanel && "bg-accent/60 text-foreground",
              )}
              onClick={toggleContents}
              aria-expanded={isDesktopContents ? contentsOpen : undefined}
              aria-label={isMobile ? "Contents" : undefined}
            >
              <ListTree className="h-4 w-4" aria-hidden="true" />
              {isMobile ? null : "Contents"}
            </Button>

            {isMobile ? null : (
              <Popover open={detailsOpen} onOpenChange={setDetailsOpen}>
                <PopoverTrigger asChild>
                  <Button type="button" variant="ghost" size="sm" className={labeledButtonClass}>
                    <Info className="h-4 w-4" aria-hidden="true" />
                    Details
                  </Button>
                </PopoverTrigger>
                <PopoverContent
                  align="end"
                  className="max-h-[min(70vh,36rem)] w-[min(24rem,calc(100vw-2rem))] overflow-y-auto"
                  aria-label="Deal details"
                >
                  {details}
                </PopoverContent>
              </Popover>
            )}

            <StarButton
              itemType="agreement"
              itemUuid={agreementUuid}
              context={{
                target: agreement.target,
                acquirer: agreement.acquirer,
                filing_date: agreement.filing_date,
              }}
              className="shrink-0"
              buttonClassName="h-10 w-10"
              showTagSwatches={false}
            />

            <DropdownMenu modal={false}>
              <DropdownMenuTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className={cn(iconButtonClass, "-mr-2")}
                  aria-label="More options"
                >
                  <MoreHorizontal aria-hidden="true" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-52">
                {isMobile ? (
                  <>
                    <DropdownMenuItem onSelect={() => setDetailsSheetOpen(true)}>
                      <Info className="mr-2 h-4 w-4" aria-hidden="true" />
                      Deal details
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                  </>
                ) : null}
                <DropdownMenuCheckboxItem
                  checked={showBodyOnly}
                  onCheckedChange={(checked) => setShowBodyOnly(checked === true)}
                  onSelect={(event) => event.preventDefault()}
                >
                  Show body only
                </DropdownMenuCheckboxItem>
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Text size</DropdownMenuLabel>
                <DropdownMenuRadioGroup
                  value={textSize}
                  onValueChange={(value) => setTextSize(value as AgreementTextSize)}
                >
                  <DropdownMenuRadioItem value="small">Small</DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="medium">Medium</DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="large">Large</DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
                <DropdownMenuSeparator />
                {overlay ? (
                  <>
                    <DropdownMenuLabel>Defined terms</DropdownMenuLabel>
                    <DropdownMenuRadioGroup
                      value={termMode}
                      onValueChange={(value) => setTermMode(value as TermHighlightMode)}
                    >
                      <DropdownMenuRadioItem value="distinctive">Underline</DropdownMenuRadioItem>
                      <DropdownMenuRadioItem value="all">
                        Underline all, incl. common words
                      </DropdownMenuRadioItem>
                      <DropdownMenuRadioItem value="off">Off</DropdownMenuRadioItem>
                    </DropdownMenuRadioGroup>
                    <DropdownMenuSeparator />
                  </>
                ) : null}
                <DropdownMenuItem
                  onSelect={() => {
                    const href = `${window.location.origin}${buildAgreementHref({ agreementUuid })}`;
                    void copy(href).then((ok) =>
                      toast(
                        ok
                          ? { title: "Link copied" }
                          : { title: "Couldn't copy the link", variant: "destructive" },
                      ),
                    );
                  }}
                >
                  Copy link to agreement
                </DropdownMenuItem>
                {agreement.url ? (
                  <DropdownMenuItem asChild>
                    <ExternalLink href={agreement.url} className="flex w-full">
                      View SEC filing
                    </ExternalLink>
                  </DropdownMenuItem>
                ) : null}
              </DropdownMenuContent>
            </DropdownMenu>

            {findOpen && isMobile ? (
              <div className="absolute inset-y-0 -inset-x-1 flex items-center bg-background px-1">
                {findBar}
              </div>
            ) : null}
          </div>

          {/* The row is kept while there is an index, even outside any
              section, so the sticky header doesn't change height mid-scroll. */}
          {overlay ? (
            <nav
              aria-label="Current section"
              className="-mt-1 hidden h-6 min-w-0 text-xs text-muted-foreground md:block"
            >
              <ol className="flex min-w-0 items-center gap-1.5">
                {(breadcrumb ?? []).map((item, index) => (
                  <li key={item.id} className="flex min-w-0 items-center gap-1.5">
                    {index > 0 ? <span aria-hidden="true">›</span> : null}
                    <button
                      type="button"
                      className="min-w-0 truncate rounded-sm hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      onClick={() => jumpToOutline(item.id)}
                      aria-current={
                        index === (breadcrumb?.length ?? 0) - 1 ? "location" : undefined
                      }
                    >
                      {item.text}
                    </button>
                  </li>
                ))}
              </ol>
            </nav>
          ) : null}

          {/* Match strip: under the header on md+, a floating bar on phones so
              the sticky chrome stays one row. */}
          {matchStrip ? (
            <div className="max-md:fixed max-md:inset-x-3 max-md:bottom-3 max-md:z-30 md:flex md:pb-2">
              {matchStrip}
            </div>
          ) : null}
        </div>
      </header>

      <section
        aria-labelledby="agreement-reader-title"
        className={cn(
          "app-frame py-5 sm:py-6",
          "print:block print:max-w-none print:px-0 print:py-0",
          showContentsPanel && "grid grid-cols-[300px_minmax(0,1fr)] gap-8",
        )}
      >
        {showContentsPanel ? (
          <aside
            aria-label="Contents"
            className="sticky self-start overflow-y-auto rounded-xl border border-border bg-card shadow-raised print:hidden print:shadow-none"
            style={{
              top: `${stickyBottom + 12}px`,
              maxHeight: `calc(100vh - ${stickyBottom + 24}px)`,
            }}
          >
            <div className="sticky top-0 z-10 flex items-center gap-2 border-b border-border bg-card px-3 py-2">
              <ListTree className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              <h2 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Contents
              </h2>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="ml-auto h-8 w-8 text-muted-foreground hover:text-foreground"
                onClick={() => setContentsOpen(false)}
                aria-label="Hide contents"
              >
                <X aria-hidden="true" />
              </Button>
            </div>
            <div className="py-2">{contents}</div>
          </aside>
        ) : null}

        <div className="min-w-0">
          {/* One measure for facts, notice and text. Next to the contents
              panel the text keeps at most a 6rem gap from it, so a wide frame
              leaves its spare width on the right instead of a gutter. */}
          <div
            className={cn(
              "w-full max-w-[calc(72ch+4rem)]",
              showContentsPanel
                ? "ml-[min(6rem,max(0px,calc((100%-72ch-4rem)/2)))]"
                : "mx-auto",
            )}
          >
            <div className="mb-4 space-y-1.5 print:hidden">
              {showAcquirer ? (
                <p className="text-sm text-muted-foreground lg:hidden">
                  Acquired by {acquirerName}
                </p>
              ) : null}
              {facts.length > 0 || agreement.url ? (
                <p className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-sm text-muted-foreground">
                  {facts.map((fact, index) => (
                    <span key={fact.key} className="inline-flex items-center gap-2">
                      {index > 0 ? <span aria-hidden="true">·</span> : null}
                      <span title={fact.label}>
                        <span className="sr-only">{fact.label}: </span>
                        {fact.value}
                      </span>
                    </span>
                  ))}
                  {agreement.url ? (
                    <span className="inline-flex items-center gap-2">
                      {facts.length > 0 ? <span aria-hidden="true">·</span> : null}
                      <ExternalLink
                        href={agreement.url}
                        className="text-primary underline-offset-4 hover:underline"
                      >
                        SEC filing
                      </ExternalLink>
                    </span>
                  ) : null}
                </p>
              ) : null}
              {agreement.is_redacted ? (
                <p className="flex items-start gap-1.5 text-xs text-warning">
                  <FileSearch className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                  <span>
                    {focusSectionUuid
                      ? "Full text is limited in anonymous mode; text around your matched section may be redacted."
                      : "Full text is limited in anonymous mode; some text may be redacted."}{" "}
                    <Link
                      to={buildAccountPathWithNext(
                        `${window.location.pathname}${window.location.search}`,
                      )}
                      className="font-medium underline underline-offset-2"
                    >
                      Sign in
                    </Link>
                  </span>
                </p>
              ) : null}
            </div>

            <article
              className="rounded-xl border border-border bg-card shadow-raised p-4 sm:p-8 print:rounded-none print:border-none print:p-0 print:shadow-none"
              aria-labelledby="agreement-reader-title"
              aria-busy={!docReady}
            >
              <div ref={contentRef} className="agreement-reader-content min-w-0">
                {docReady && renderedXml !== null ? (
                  <XMLRenderer
                    xmlContent={renderedXml}
                    mode="agreement"
                    isMobile={isMobile}
                    className={cn(
                      "leading-relaxed text-foreground break-words",
                      agreementTextSizeClass,
                    )}
                    showBodyOnly={showBodyOnly}
                    findQuery={find.appliedQuery}
                    matchedSectionUuids={matchedSet}
                    currentMatchUuid={currentMatchUuid}
                    revealRequest={revealRequest}
                    renderSectionActions={renderSectionActions}
                    readerIndex={readerIndex}
                    termMode={termMode}
                    onOverlayChange={setOverlay}
                  />
                ) : (
                  <div role="status" className="space-y-3">
                    <span className="sr-only">Loading agreement text</span>
                    <Skeleton className="h-5 w-1/2" />
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-11/12" />
                    <Skeleton className="h-4 w-4/5" />
                  </div>
                )}
              </div>
            </article>
          </div>
        </div>
      </section>

      <Sheet open={tocSheetOpen} onOpenChange={setTocSheetOpen}>
        <SheetContent side="left" className="w-[min(340px,100vw)] overflow-y-auto p-0">
          <div className="sticky top-0 border-b border-border bg-background py-3 pl-4 pr-12">
            <SheetTitle className="text-base">Contents</SheetTitle>
            <SheetDescription className="sr-only">
              Browse agreement sections and jump to one.
            </SheetDescription>
          </div>
          <div className="py-2">{contents}</div>
        </SheetContent>
      </Sheet>

      {overlay && docReady ? (
        <ReaderOverlayLayer
          overlay={overlay}
          containerRef={contentRef}
          isMobile={isMobile}
          onFollowXref={followXref}
          onGoToAnchor={(key, container) => scrollToTextAnchor(key, container)}
        />
      ) : null}

      {overlay && historyState?.readerJump ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className={cn(
            "fixed left-1/2 z-30 -translate-x-1/2 gap-1.5 rounded-full bg-background shadow-overlay print:hidden",
            showMatchStrip ? "max-md:bottom-20 md:bottom-6" : "bottom-6",
          )}
          onClick={() => navigate(-1)}
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          {historyState.readerBack ? `Back to ${historyState.readerBack}` : "Back"}
        </Button>
      ) : null}

      <Sheet open={detailsSheetOpen} onOpenChange={setDetailsSheetOpen}>
        <SheetContent side="bottom" className="max-h-[85vh] overflow-y-auto">
          <SheetTitle className="mb-2 text-base">Deal details</SheetTitle>
          <SheetDescription className="sr-only">Metadata for this agreement.</SheetDescription>
          {details}
        </SheetContent>
      </Sheet>
    </div>
  );
}
