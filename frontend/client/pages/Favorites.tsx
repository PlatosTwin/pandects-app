import { CSS } from "@dnd-kit/utilities";
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import { sortableKeyboardCoordinates } from "@dnd-kit/sortable";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, Navigate, useLocation } from "react-router-dom";
import {
  Check,
  ChevronDown,
  Folder,
  GripVertical,
  Plus,
  ExternalLink,
  Filter,
  Pencil,
  Star,
  Trash2,
  X,
} from "lucide-react";

import { PageShell } from "@/components/PageShell";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TagPill, TagSwatch } from "@/components/favorites/TagPill";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { useFavorites } from "@/contexts/FavoritesContext";
import { authFetch } from "@/lib/auth-fetch";
import { apiUrl } from "@/lib/api-config";
import { formatCompactCurrencyValue } from "@/lib/format-utils";
import { indexClauseTypeLabels } from "@/lib/clause-type-index";
import { useFilterOptions } from "@/hooks/use-filter-options";
import {
  bulkCopyFavorites as apiBulkCopyFavorites,
  bulkMoveFavorites as apiBulkMoveFavorites,
  deleteFavorite as apiDeleteFavorite,
  listFavorites,
  patchFavorite as apiPatchFavorite,
  setFavoriteTags as apiSetFavoriteTags,
  TAG_COLORS,
  type Favorite,
  type FavoriteItemType,
  type FavoriteProject,
  type FavoriteTag,
  type TagColor,
} from "@/lib/favorites-api";
import type { Agreement } from "@shared/agreement";
import { TagEditor } from "@/components/favorites/TagEditor";

type Filter = "all" | FavoriteItemType;

const TYPE_LABELS: Record<FavoriteItemType, string> = {
  section: "Section",
  agreement: "Deal",
  tax_clause: "Tax clause",
};

interface FavoriteFilters {
  tagIds: string[];
  yearMin: string;
  yearMax: string;
  sizeMinUsd: string;
  sizeMaxUsd: string;
  target: string;
  acquirer: string;
}

interface SectionDetails {
  agreement_uuid: string | null;
  section_uuid: string;
  section_standard_id: string[];
  xml: string | null;
  article_title: string | null;
  section_title: string | null;
}

const EMPTY_FILTERS: FavoriteFilters = {
  tagIds: [],
  yearMin: "",
  yearMax: "",
  sizeMinUsd: "",
  sizeMaxUsd: "",
  target: "",
  acquirer: "",
};

const SIZE_OPTIONS = [
  { label: "$50M", value: "50000000" },
  { label: "$100M", value: "100000000" },
  { label: "$250M", value: "250000000" },
  { label: "$500M", value: "500000000" },
  { label: "$1B", value: "1000000000" },
  { label: "$2.5B", value: "2500000000" },
  { label: "$5B", value: "5000000000" },
  { label: "$10B", value: "10000000000" },
];

const ANY_SIZE_VALUE = "any";

function hasActiveFilters(f: FavoriteFilters): boolean {
  return (
    f.tagIds.length > 0 ||
    f.yearMin.trim() !== "" ||
    f.yearMax.trim() !== "" ||
    f.sizeMinUsd.trim() !== "" ||
    f.sizeMaxUsd.trim() !== "" ||
    f.target.trim() !== "" ||
    f.acquirer.trim() !== ""
  );
}

function favoriteHref(fav: Favorite): string | null {
  const agreementUuid = fav.agreement_uuid;
  if (!agreementUuid) return null;
  if (fav.item_type === "section") {
    return `/agreements/${agreementUuid}?focusSectionUuid=${fav.item_uuid}`;
  }
  return `/agreements/${agreementUuid}`;
}

function favoriteHeading(fav: Favorite, agreement: Agreement | null): string {
  const ctxTarget =
    agreement?.target ??
    (typeof fav.context?.target === "string"
      ? (fav.context.target as string)
      : null);
  const ctxAcquirer =
    agreement?.acquirer ??
    (typeof fav.context?.acquirer === "string"
      ? (fav.context.acquirer as string)
      : null);
  if (ctxTarget && ctxAcquirer) return `${ctxTarget} — ${ctxAcquirer}`;
  if (ctxTarget || ctxAcquirer) return ctxTarget ?? ctxAcquirer ?? "";
  return fav.item_uuid;
}

function contextString(fav: Favorite, key: string): string | null {
  const value = fav.context?.[key];
  return typeof value === "string" && value.trim() ? value : null;
}

function stripXmlText(xml: string | null | undefined): string {
  if (!xml) return "";
  if (typeof DOMParser !== "undefined") {
    const parsed = new DOMParser().parseFromString(xml, "text/xml");
    const parseError = parsed.querySelector("parsererror");
    if (!parseError) return parsed.documentElement.textContent ?? "";
  }
  return xml.replace(/<[^>]*>/g, " ");
}

function firstWords(text: string, count: number): string {
  const words = text.replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
  if (words.length <= count) return words.join(" ");
  return `${words.slice(0, count).join(" ")}…`;
}

function formatDate(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString();
}

function FavoriteRow({
  fav,
  agreement,
  sectionDetails,
  clauseTypeLabelById,
  selected,
  onSelectChange,
  onUpdated,
  onRemoved,
  onTagsChanged,
  onTagDeleted,
}: {
  fav: Favorite;
  agreement: Agreement | null;
  sectionDetails: SectionDetails | null;
  clauseTypeLabelById: Record<string, string>;
  selected: boolean;
  onSelectChange: (id: string, selected: boolean) => void;
  onUpdated: (next: Favorite) => void;
  onRemoved: (id: string) => void;
  onTagsChanged: (id: string, tags: FavoriteTag[]) => void;
  onTagDeleted: (tagId: string) => void;
}) {
  const { toast } = useToast();
  const [editingNote, setEditingNote] = useState(false);
  const [noteDraft, setNoteDraft] = useState(fav.note ?? "");
  const [busy, setBusy] = useState(false);

  const href = favoriteHref(fav);
  const heading = favoriteHeading(fav, agreement);

  const handleSaveNote = useCallback(async () => {
    setBusy(true);
    try {
      const { favorite } = await apiPatchFavorite(fav.id, {
        note: noteDraft.trim() || null,
      });
      onUpdated(favorite);
      setEditingNote(false);
    } catch {
      toast({ title: "Couldn't save note", variant: "destructive" });
    } finally {
      setBusy(false);
    }
  }, [fav.id, noteDraft, onUpdated, toast]);

  const handleRemove = useCallback(async () => {
    setBusy(true);
    try {
      await apiDeleteFavorite(fav.id);
      onRemoved(fav.id);
    } catch {
      toast({ title: "Couldn't remove favorite", variant: "destructive" });
    } finally {
      setBusy(false);
    }
  }, [fav.id, onRemoved, toast]);

  const handleChangeTags = useCallback(
    async (nextTagIds: string[]) => {
      setBusy(true);
      try {
        const fresh = await apiSetFavoriteTags(fav.id, nextTagIds);
        onTagsChanged(fav.id, fresh);
      } catch {
        toast({ title: "Couldn't update tags", variant: "destructive" });
      } finally {
        setBusy(false);
      }
    },
    [fav.id, onTagsChanged, toast],
  );

  const filingYear = agreement?.year ?? null;
  const dealValue = agreement?.transaction_price_total ?? null;
  const sectionArticleTitle =
    sectionDetails?.article_title ?? contextString(fav, "article_title");
  const sectionTitle =
    sectionDetails?.section_title ?? contextString(fav, "section_title");
  const sectionStandardIds =
    sectionDetails?.section_standard_id ??
    (Array.isArray(fav.context?.standard_id)
      ? fav.context.standard_id.filter(
          (value): value is string => typeof value === "string",
        )
      : []);
  const sectionSnippet = firstWords(
    stripXmlText(sectionDetails?.xml ?? contextString(fav, "xml")),
    25,
  );
  const { attributes, listeners, setNodeRef, transform, isDragging } =
    useDraggable({
      id: fav.id,
      data: { favoriteId: fav.id, projectId: fav.project_id },
    });
  const style = {
    transform: CSS.Translate.toString(transform),
  };

  return (
    <Card
      ref={setNodeRef}
      style={style}
      className={`relative overflow-hidden border-l-4 border-l-amber-400 bg-card shadow-sm transition-shadow hover:shadow-md ${isDragging ? "z-10 opacity-80 ring-2 ring-primary" : ""}`}
    >
      <div className="relative p-4">
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-1.5 sm:pr-48">
            <button
              type="button"
              className="inline-flex h-10 w-10 cursor-grab items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground active:cursor-grabbing sm:h-6 sm:w-6"
              aria-label="Drag favorite to project"
              title="Drag to project"
              {...attributes}
              {...listeners}
            >
              <GripVertical className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
            <Checkbox
              checked={selected}
              onCheckedChange={(checked) =>
                onSelectChange(fav.id, checked === true)
              }
              aria-label={`Select ${heading}`}
              className="mr-1"
            />
            <Badge
              variant="secondary"
              className="rounded-md px-1.5 py-0 text-[11px]"
            >
              {TYPE_LABELS[fav.item_type]}
            </Badge>
            <span className="text-xs text-muted-foreground">
              Starred {formatDate(fav.created_at)}
            </span>
            {filingYear ? (
              <span className="text-xs text-muted-foreground">
                {filingYear}
              </span>
            ) : null}
            {dealValue ? (
              <span className="text-xs text-muted-foreground">
                {formatCompactCurrencyValue(dealValue)}
              </span>
            ) : null}
          </div>
          <div className="text-base font-semibold leading-snug text-foreground sm:pr-48">
            {heading}
          </div>

          {fav.item_type === "section" ? (
            <div className="space-y-2 rounded-md border bg-muted/20 p-3">
              {sectionStandardIds.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {sectionStandardIds.map((id) => (
                    <Badge
                      key={id}
                      variant="outline"
                      className="max-w-full rounded-md px-2 py-0.5 text-[11px] font-medium"
                      title={clauseTypeLabelById[id] ?? id}
                    >
                      <span className="truncate">
                        {clauseTypeLabelById[id] ?? id}
                      </span>
                    </Badge>
                  ))}
                </div>
              ) : null}
              {(sectionArticleTitle || sectionTitle) && (
                <div className="text-sm leading-snug text-foreground">
                  {sectionArticleTitle ? (
                    <span className="font-medium">{sectionArticleTitle}</span>
                  ) : null}
                  {sectionArticleTitle && sectionTitle ? (
                    <span className="text-muted-foreground"> / </span>
                  ) : null}
                  {sectionTitle ? <span>{sectionTitle}</span> : null}
                </div>
              )}
              {sectionSnippet ? (
                <p className="line-clamp-2 text-sm leading-relaxed text-muted-foreground">
                  {sectionSnippet}
                </p>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Section details are loading…
                </p>
              )}
            </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-1.5">
            <TagEditor
              selectedTagIds={fav.tags.map((t) => t.id)}
              onChange={(nextTagIds) => void handleChangeTags(nextTagIds)}
              onTagDeleted={onTagDeleted}
              showSelectedTags={false}
              triggerLabel={fav.tags.length === 0 ? "Add tags" : "Edit tags"}
            />
            {fav.tags.map((t) => (
              <TagPill
                key={t.id}
                name={t.name}
                color={t.color}
                onRemove={() =>
                  void handleChangeTags(
                    fav.tags
                      .filter((tag) => tag.id !== t.id)
                      .map((tag) => tag.id),
                  )
                }
              />
            ))}
          </div>

          {editingNote ? (
            <div className="space-y-2">
              <Textarea
                autoFocus
                aria-label={`Note for ${heading}`}
                value={noteDraft}
                onChange={(e) => setNoteDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (!busy) void handleSaveNote();
                  }
                }}
                rows={3}
                placeholder="Add a note… (Enter to save, Shift+Enter for newline)"
              />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  onClick={() => void handleSaveNote()}
                  disabled={busy}
                >
                  {busy ? "Saving…" : "Save"}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setEditingNote(false);
                    setNoteDraft(fav.note ?? "");
                  }}
                  disabled={busy}
                >
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setEditingNote(true)}
              className="block min-h-11 max-w-full whitespace-pre-wrap break-words text-left text-sm text-muted-foreground hover:text-foreground sm:min-h-0"
            >
              {fav.note ?? "Add a note…"}
            </button>
          )}
        </div>
        <div className="mt-3 flex shrink-0 gap-2 sm:absolute sm:right-4 sm:top-4 sm:mt-0">
          {href ? (
            <Button asChild variant="outline" size="sm" className="gap-1.5">
              <Link to={href}>
                <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                Open
              </Link>
            </Button>
          ) : null}
          <Button
            variant="ghost"
            size="icon"
            aria-label="Remove favorite"
            title="Remove favorite"
            onClick={() => void handleRemove()}
            disabled={busy}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </Card>
  );
}

function FilterBar({
  filters,
  onChange,
  onClear,
}: {
  filters: FavoriteFilters;
  onChange: (next: FavoriteFilters) => void;
  onClear: () => void;
}) {
  const { tags, ensureTagsLoaded } = useFavorites();
  const [open, setOpen] = useState(false);
  useEffect(() => {
    ensureTagsLoaded();
  }, [ensureTagsLoaded]);

  const toggleTag = (tagId: string) => {
    if (filters.tagIds.includes(tagId)) {
      onChange({
        ...filters,
        tagIds: filters.tagIds.filter((id) => id !== tagId),
      });
    } else {
      onChange({ ...filters, tagIds: [...filters.tagIds, tagId] });
    }
  };

  const active = hasActiveFilters(filters);
  const setSize = (key: "sizeMinUsd" | "sizeMaxUsd", value: string) => {
    onChange({ ...filters, [key]: value === ANY_SIZE_VALUE ? "" : value });
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="grid min-h-12 w-full grid-cols-[1fr_auto] items-center gap-2 rounded-lg border border-border bg-background/80 px-3 py-2 text-left shadow-sm transition-colors hover:bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={open ? "Close filters" : "Open filters"}
        >
          <span className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <Filter className="h-3.5 w-3.5" aria-hidden="true" />
            Filters
            {active ? (
              <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-foreground">
                Active
              </span>
            ) : null}
          </span>
          <span className="inline-flex h-7 items-center gap-1 rounded-md border px-2 text-xs font-medium text-foreground">
            <ChevronDown
              className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
              aria-hidden="true"
            />
            {open ? "Hide" : "Show"}
          </span>
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        sideOffset={8}
        className="w-[min(calc(100vw-2rem),72rem)] p-0"
      >
        <div className="px-4 py-3">
          {active ? (
            <div className="mb-2 flex justify-end">
              <Button
                variant="ghost"
                size="sm"
                className="h-11 gap-1 px-3 text-xs sm:h-7 sm:px-2"
                onClick={onClear}
              >
                <X className="h-3 w-3" /> Clear
              </Button>
            </div>
          ) : null}
          <div className="space-y-3">
            {tags.length > 0 ? (
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">Tags</div>
                <div className="flex flex-wrap gap-1.5">
                  {tags.map((t) => (
                    <TagPill
                      key={t.id}
                      name={t.name}
                      color={t.color}
                      selected={filters.tagIds.includes(t.id)}
                      onClick={() => toggleTag(t.id)}
                    />
                  ))}
                </div>
              </div>
            ) : null}
            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_16rem_minmax(20rem,24rem)]">
              <label className="space-y-1 text-xs">
                <div className="text-muted-foreground">Target</div>
                <Input
                  aria-label="Target filter"
                  value={filters.target}
                  onChange={(e) =>
                    onChange({ ...filters, target: e.target.value })
                  }
                  placeholder="e.g. Acme"
                  className="h-11 text-sm sm:h-8"
                />
              </label>
              <label className="space-y-1 text-xs">
                <div className="text-muted-foreground">Acquirer</div>
                <Input
                  aria-label="Acquirer filter"
                  value={filters.acquirer}
                  onChange={(e) =>
                    onChange({ ...filters, acquirer: e.target.value })
                  }
                  placeholder="e.g. Globex"
                  className="h-11 text-sm sm:h-8"
                />
              </label>
              <div className="space-y-1 text-xs">
                <div className="text-muted-foreground">Year</div>
                <div className="flex items-center gap-1">
                  <Input
                    aria-label="Minimum year"
                    type="number"
                    value={filters.yearMin}
                    onChange={(e) =>
                      onChange({ ...filters, yearMin: e.target.value })
                    }
                    placeholder="From"
                    className="h-11 text-sm sm:h-8"
                  />
                  <span aria-hidden="true">-</span>
                  <Input
                    aria-label="Maximum year"
                    type="number"
                    value={filters.yearMax}
                    onChange={(e) =>
                      onChange({ ...filters, yearMax: e.target.value })
                    }
                    placeholder="To"
                    className="h-11 text-sm sm:h-8"
                  />
                </div>
              </div>
              <div className="space-y-1 text-xs">
                <div className="text-muted-foreground">
                  Transaction size (USD)
                </div>
                <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-1">
                  <Select
                    value={filters.sizeMinUsd || ANY_SIZE_VALUE}
                    onValueChange={(value) => setSize("sizeMinUsd", value)}
                  >
                    <SelectTrigger
                      className="h-11 text-sm sm:h-8"
                      aria-label="Minimum transaction size"
                    >
                      <SelectValue placeholder="Min" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={ANY_SIZE_VALUE}>Any min</SelectItem>
                      {SIZE_OPTIONS.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <span aria-hidden="true">-</span>
                  <Select
                    value={filters.sizeMaxUsd || ANY_SIZE_VALUE}
                    onValueChange={(value) => setSize("sizeMaxUsd", value)}
                  >
                    <SelectTrigger
                      className="h-11 text-sm sm:h-8"
                      aria-label="Maximum transaction size"
                    >
                      <SelectValue placeholder="Max" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={ANY_SIZE_VALUE}>Any max</SelectItem>
                      {SIZE_OPTIONS.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

function TagsManager({
  onTagUpdated,
  onTagDeleted,
}: {
  onTagUpdated: (tag: FavoriteTag) => void;
  onTagDeleted: (tagId: string) => void;
}) {
  const { toast } = useToast();
  const { tags, ensureTagsLoaded, updateTag, removeTag } = useFavorites();
  const [open, setOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [draftColor, setDraftColor] = useState<TagColor>("blue");
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    ensureTagsLoaded();
  }, [ensureTagsLoaded]);

  const beginEdit = (tag: FavoriteTag) => {
    setEditingId(tag.id);
    setDraftName(tag.name);
    setDraftColor(tag.color);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setDraftName("");
    setDraftColor("blue");
  };

  const handleSave = async (tag: FavoriteTag) => {
    const name = draftName.trim();
    if (!name) return;
    setBusyId(tag.id);
    try {
      const updated = await updateTag(tag.id, { name, color: draftColor });
      onTagUpdated(updated);
      cancelEdit();
    } catch {
      toast({ title: "Couldn't update tag", variant: "destructive" });
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (tag: FavoriteTag) => {
    const confirmed = window.confirm(
      `Delete "${tag.name}" permanently? This removes it from every favorite.`,
    );
    if (!confirmed) return;
    setBusyId(tag.id);
    try {
      await removeTag(tag.id);
      onTagDeleted(tag.id);
      if (editingId === tag.id) cancelEdit();
    } catch {
      toast({ title: "Couldn't delete tag", variant: "destructive" });
    } finally {
      setBusyId(null);
    }
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="grid min-h-12 w-full grid-cols-[1fr_auto] items-center gap-2 rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-left shadow-sm transition-colors hover:bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={open ? "Close tag manager" : "Open tag manager"}
        >
          <span className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Tags
            <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-foreground">
              {tags.length}
            </span>
          </span>
          <span className="inline-flex h-7 items-center gap-1 rounded-md px-2 text-xs font-medium text-muted-foreground">
            <ChevronDown
              className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
              aria-hidden="true"
            />
            Manage
          </span>
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={8}
        className="w-[min(calc(100vw-2rem),42rem)] p-0"
      >
        <div className="p-3">
          {tags.length === 0 ? (
            <div className="text-sm text-muted-foreground">No tags yet.</div>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {tags.map((tag) => {
                const editing = editingId === tag.id;
                const busy = busyId === tag.id;
                return (
                  <div
                    key={tag.id}
                    className="rounded-md border bg-background p-2"
                  >
                    {editing ? (
                      <div className="space-y-2">
                        <Input
                          aria-label={`Tag name for ${tag.name}`}
                          value={draftName}
                          onChange={(e) => setDraftName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !busy) {
                              e.preventDefault();
                              void handleSave(tag);
                            }
                          }}
                          className="h-11 text-sm sm:h-8"
                          autoFocus
                        />
                        <div className="flex flex-wrap gap-1.5">
                          {TAG_COLORS.map((color) => (
                            <button
                              key={color}
                              type="button"
                              onClick={() => setDraftColor(color)}
                              aria-label={`Use ${color} color`}
                              aria-pressed={draftColor === color}
                              className={
                                "relative inline-grid h-11 w-11 place-items-center rounded-full ring-1 ring-transparent transition-shadow sm:h-6 sm:w-6 " +
                                (draftColor === color
                                  ? "ring-2 ring-foreground ring-offset-2 ring-offset-background"
                                  : "hover:ring-muted-foreground/40")
                              }
                            >
                              <TagSwatch color={color} className="h-4 w-4" />
                              {draftColor === color ? (
                                <Check className="pointer-events-none absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 text-white drop-shadow" />
                              ) : null}
                            </button>
                          ))}
                        </div>
                        <div className="flex justify-end gap-1">
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-11 px-3 text-xs sm:h-7 sm:px-2"
                            onClick={cancelEdit}
                            disabled={busy}
                          >
                            Cancel
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            className="h-11 px-3 text-xs sm:h-7 sm:px-2"
                            onClick={() => void handleSave(tag)}
                            disabled={busy || !draftName.trim()}
                          >
                            {busy ? "Saving..." : "Save"}
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-between gap-2">
                        <TagPill name={tag.name} color={tag.color} />
                        <div className="flex shrink-0 gap-1">
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="h-11 w-11 sm:h-7 sm:w-7"
                            aria-label={`Edit tag ${tag.name}`}
                            title={`Edit tag ${tag.name}`}
                            onClick={() => beginEdit(tag)}
                            disabled={busy}
                          >
                            <Pencil className="h-3.5 w-3.5" />
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="h-11 w-11 text-muted-foreground hover:text-destructive sm:h-7 sm:w-7"
                            aria-label={`Delete tag ${tag.name}`}
                            title={`Delete tag ${tag.name}`}
                            onClick={() => void handleDelete(tag)}
                            disabled={busy}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}

function ProjectDropButton({
  project,
  active,
  count,
  onSelect,
}: {
  project: FavoriteProject;
  active: boolean;
  count: number;
  onSelect: (projectId: string) => void;
}) {
  const { isOver, setNodeRef } = useDroppable({
    id: `project:${project.id}`,
    data: { projectId: project.id },
  });
  return (
    <button
      ref={setNodeRef}
      type="button"
      onClick={() => onSelect(project.id)}
      aria-pressed={active}
      className={`flex w-full items-center justify-between gap-2 rounded-md border px-2.5 py-2 text-left text-sm transition-colors ${
        active
          ? "border-primary bg-primary/10 text-foreground"
          : "border-transparent bg-background hover:border-border hover:bg-muted/60"
      } ${isOver ? "ring-2 ring-primary ring-offset-1 ring-offset-background" : ""}`}
    >
      <span className="flex min-w-0 items-center gap-2">
        <TagSwatch color={project.color} className="h-3 w-3 shrink-0" />
        <span className="truncate font-medium">{project.name}</span>
      </span>
      <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
        {count}
      </span>
    </button>
  );
}

function ProjectSidebar({
  projects,
  activeProjectId,
  projectCounts,
  onSelectProject,
  onProjectCreated,
  onProjectUpdated,
  onProjectDeleted,
}: {
  projects: FavoriteProject[];
  activeProjectId: string | null;
  projectCounts: Record<string, number>;
  onSelectProject: (projectId: string | null) => void;
  onProjectCreated: (project: FavoriteProject) => void;
  onProjectUpdated: (project: FavoriteProject) => void;
  onProjectDeleted: (projectId: string, reassignProjectId: string) => void;
}) {
  const { toast } = useToast();
  const { createProject, updateProject, removeProject } = useFavorites();
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState<TagColor>("blue");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [draftColor, setDraftColor] = useState<TagColor>("blue");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(true);

  const total = Object.values(projectCounts).reduce(
    (sum, count) => sum + count,
    0,
  );

  const beginEdit = (project: FavoriteProject) => {
    setEditingId(project.id);
    setDraftName(project.name);
    setDraftColor(project.color);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setDraftName("");
    setDraftColor("blue");
  };

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    try {
      const project = await createProject(name, newColor);
      onProjectCreated(project);
      setNewName("");
      setNewColor("blue");
    } catch {
      toast({ title: "Couldn't create project", variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  const handleSave = async (project: FavoriteProject) => {
    const name = draftName.trim();
    if (!name) return;
    setBusy(true);
    try {
      const updated = await updateProject(project.id, {
        name,
        color: draftColor,
      });
      onProjectUpdated(updated);
      cancelEdit();
    } catch {
      toast({ title: "Couldn't update project", variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (project: FavoriteProject) => {
    const target = projects.find((candidate) => candidate.id !== project.id);
    if (!target) {
      toast({
        title: "Create another project first",
        description:
          "Favorites need a destination before this project can be deleted.",
        variant: "destructive",
      });
      return;
    }
    const confirmed = window.confirm(
      `Delete "${project.name}" and move its favorites to "${target.name}"?`,
    );
    if (!confirmed) return;
    setBusy(true);
    try {
      const result = await removeProject(project.id, target.id);
      onProjectDeleted(project.id, result.reassigned_to_project_id);
      if (activeProjectId === project.id) {
        onSelectProject(result.reassigned_to_project_id);
      }
    } catch {
      toast({ title: "Couldn't delete project", variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside className="self-start rounded-lg border bg-background p-3 shadow-sm lg:sticky lg:top-20">
      <Collapsible open={open} onOpenChange={setOpen}>
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <Folder className="h-3.5 w-3.5" aria-hidden="true" />
            Projects
          </div>
          <div className="flex items-center gap-1">
            <Badge variant="secondary">{projects.length}</Badge>
            <CollapsibleTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-11 w-11 sm:h-7 sm:w-7"
                aria-label={open ? "Collapse projects" : "Expand projects"}
              >
                <ChevronDown
                  className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
                  aria-hidden="true"
                />
              </Button>
            </CollapsibleTrigger>
          </div>
        </div>

        <CollapsibleContent className="mt-3 space-y-3">
          <div className="space-y-1.5">
            <button
              type="button"
              onClick={() => onSelectProject(null)}
              aria-pressed={activeProjectId === null}
              className={`flex w-full items-center justify-between gap-2 rounded-md border px-2.5 py-2 text-left text-sm transition-colors ${
                activeProjectId === null
                  ? "border-primary bg-primary/10 text-foreground"
                  : "border-transparent bg-background hover:border-border hover:bg-muted/60"
              }`}
            >
              <span className="font-medium">All projects</span>
              <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                {total}
              </span>
            </button>
            {projects.map((project) => (
              <div key={project.id} className="rounded-md">
                {editingId === project.id ? (
                  <div className="space-y-2 rounded-md border p-2">
                    <Input
                      aria-label={`Project name for ${project.name}`}
                      value={draftName}
                      onChange={(e) => setDraftName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !busy) {
                          e.preventDefault();
                          void handleSave(project);
                        }
                      }}
                      className="h-11 text-sm sm:h-8"
                      autoFocus
                    />
                    <div className="flex flex-wrap gap-1.5">
                      {TAG_COLORS.map((color) => (
                        <button
                          key={color}
                          type="button"
                          onClick={() => setDraftColor(color)}
                          aria-label={`Use ${color} color`}
                          aria-pressed={draftColor === color}
                          className={
                            "relative inline-grid h-11 w-11 place-items-center rounded-full ring-1 ring-transparent transition-shadow sm:h-6 sm:w-6 " +
                            (draftColor === color
                              ? "ring-2 ring-foreground ring-offset-2 ring-offset-background"
                              : "hover:ring-muted-foreground/40")
                          }
                        >
                          <TagSwatch color={color} className="h-4 w-4" />
                          {draftColor === color ? (
                            <Check className="pointer-events-none absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 text-white drop-shadow" />
                          ) : null}
                        </button>
                      ))}
                    </div>
                    <div className="flex justify-end gap-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-11 px-3 text-xs sm:h-7 sm:px-2"
                        onClick={cancelEdit}
                        disabled={busy}
                      >
                        Cancel
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        className="h-11 px-3 text-xs sm:h-7 sm:px-2"
                        onClick={() => void handleSave(project)}
                        disabled={busy || !draftName.trim()}
                      >
                        Save
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-[1fr_auto] items-center gap-1">
                    <ProjectDropButton
                      project={project}
                      active={activeProjectId === project.id}
                      count={projectCounts[project.id] ?? 0}
                      onSelect={onSelectProject}
                    />
                    <div className="flex">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-11 w-11 sm:h-8 sm:w-8"
                        aria-label={`Edit project ${project.name}`}
                        title={`Edit project ${project.name}`}
                        onClick={() => beginEdit(project)}
                        disabled={busy}
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-11 w-11 text-muted-foreground hover:text-destructive sm:h-8 sm:w-8"
                        aria-label={`Delete project ${project.name}`}
                        title={`Delete project ${project.name}`}
                        onClick={() => void handleDelete(project)}
                        disabled={busy}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="space-y-2 border-t pt-3">
            <div className="text-xs font-medium text-muted-foreground">
              New project
            </div>
            <Input
              aria-label="New project name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !busy) {
                  e.preventDefault();
                  void handleCreate();
                }
              }}
              placeholder="Project name"
              className="h-11 text-sm sm:h-8"
            />
            <div className="flex flex-wrap gap-1.5">
              {TAG_COLORS.map((color) => (
                <button
                  key={color}
                  type="button"
                  onClick={() => setNewColor(color)}
                  aria-label={`Use ${color} color`}
                  aria-pressed={newColor === color}
                  className={
                    "relative inline-grid h-11 w-11 place-items-center rounded-full ring-1 ring-transparent transition-shadow sm:h-6 sm:w-6 " +
                    (newColor === color
                      ? "ring-2 ring-foreground ring-offset-2 ring-offset-background"
                      : "hover:ring-muted-foreground/40")
                  }
                >
                  <TagSwatch color={color} className="h-4 w-4" />
                  {newColor === color ? (
                    <Check className="pointer-events-none absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 text-white drop-shadow" />
                  ) : null}
                </button>
              ))}
            </div>
            <Button
              type="button"
              size="sm"
              className="h-11 w-full gap-1.5 sm:h-8"
              onClick={() => void handleCreate()}
              disabled={busy || !newName.trim()}
            >
              <Plus className="h-3.5 w-3.5" aria-hidden="true" />
              Create
            </Button>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </aside>
  );
}

function fetchAgreementMetadata(
  agreementUuid: string,
): Promise<Agreement | null> {
  return authFetch(apiUrl(`v1/agreements/${agreementUuid}`))
    .then((r) => (r.ok ? (r.json() as Promise<Agreement>) : null))
    .catch(() => null);
}

function fetchSectionDetails(
  sectionUuid: string,
): Promise<SectionDetails | null> {
  return authFetch(apiUrl(`v1/sections/${sectionUuid}`))
    .then((r) => (r.ok ? (r.json() as Promise<SectionDetails>) : null))
    .catch(() => null);
}

export default function FavoritesPage() {
  const { status, user } = useAuth();
  const location = useLocation();
  const { toast } = useToast();
  const { clause_types } = useFilterOptions({ fields: ["clause_types"] });
  const clauseTypeLabelById = useMemo(
    () => indexClauseTypeLabels(clause_types),
    [clause_types],
  );
  const {
    projects,
    ensureProjectsLoaded,
    reloadProjects,
    setFavoriteTagsCache,
  } = useFavorites();
  const [filter, setFilter] = useState<Filter>("all");
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [favorites, setFavorites] = useState<Favorite[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<FavoriteFilters>(EMPTY_FILTERS);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkProjectId, setBulkProjectId] = useState("");
  const [bulkCopyProjectId, setBulkCopyProjectId] = useState("");
  const [agreementByUuid, setAgreementByUuid] = useState<
    Record<string, Agreement | null>
  >({});
  const [sectionByUuid, setSectionByUuid] = useState<
    Record<string, SectionDetails | null>
  >({});
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );

  useEffect(() => {
    if (status !== "authenticated" || !user) return;
    ensureProjectsLoaded();
    setLoading(true);
    listFavorites()
      .then((rows) => {
        const normalized = rows.map((row) => ({
          ...row,
          project_ids:
            row.project_ids.length > 0 ? row.project_ids : [row.project_id],
        }));
        setFavorites(normalized);
        for (const fav of normalized) {
          setFavoriteTagsCache(fav.id, fav.tags);
        }
      })
      .catch(() => {
        toast({
          title: "Couldn't load favorites",
          variant: "destructive",
        });
      })
      .finally(() => setLoading(false));
  }, [status, user, toast, ensureProjectsLoaded, setFavoriteTagsCache]);

  // Lazy-fetch agreement metadata for filtering and richer display.
  useEffect(() => {
    const needed = new Set<string>();
    for (const fav of favorites) {
      const uuid = fav.agreement_uuid;
      if (uuid && agreementByUuid[uuid] === undefined) {
        needed.add(uuid);
      }
    }
    if (needed.size === 0) return;
    let cancelled = false;
    void Promise.all(
      Array.from(needed).map((uuid) =>
        fetchAgreementMetadata(uuid).then((agreement) => ({ uuid, agreement })),
      ),
    ).then((rows) => {
      if (cancelled) return;
      setAgreementByUuid((prev) => {
        const next = { ...prev };
        for (const { uuid, agreement } of rows) {
          next[uuid] = agreement;
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [favorites, agreementByUuid]);

  useEffect(() => {
    const needed = new Set<string>();
    for (const fav of favorites) {
      if (
        fav.item_type === "section" &&
        sectionByUuid[fav.item_uuid] === undefined
      ) {
        needed.add(fav.item_uuid);
      }
    }
    if (needed.size === 0) return;
    let cancelled = false;
    void Promise.all(
      Array.from(needed).map((sectionUuid) =>
        fetchSectionDetails(sectionUuid).then((section) => ({
          sectionUuid,
          section,
        })),
      ),
    ).then((rows) => {
      if (cancelled) return;
      setSectionByUuid((prev) => {
        const next = { ...prev };
        for (const { sectionUuid, section } of rows) {
          next[sectionUuid] = section;
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [favorites, sectionByUuid]);

  const counts = useMemo(() => {
    const c: Record<Filter, number> = {
      all: favorites.length,
      section: 0,
      agreement: 0,
      tax_clause: 0,
    };
    for (const fav of favorites) c[fav.item_type] += 1;
    return c;
  }, [favorites]);

  const projectCounts = useMemo(() => {
    const next: Record<string, number> = {};
    for (const fav of favorites) {
      const projectIds =
        fav.project_ids.length > 0 ? fav.project_ids : [fav.project_id];
      for (const projectId of projectIds) {
        next[projectId] = (next[projectId] ?? 0) + 1;
      }
    }
    return next;
  }, [favorites]);

  const visible = useMemo(() => {
    const yearMin = filters.yearMin.trim() ? Number(filters.yearMin) : null;
    const yearMax = filters.yearMax.trim() ? Number(filters.yearMax) : null;
    const sizeMin = filters.sizeMinUsd.trim()
      ? Number(filters.sizeMinUsd)
      : null;
    const sizeMax = filters.sizeMaxUsd.trim()
      ? Number(filters.sizeMaxUsd)
      : null;
    const targetQ = filters.target.trim().toLowerCase();
    const acquirerQ = filters.acquirer.trim().toLowerCase();

    return favorites.filter((fav) => {
      if (
        activeProjectId !== null &&
        !fav.project_ids.includes(activeProjectId) &&
        fav.project_id !== activeProjectId
      ) {
        return false;
      }
      if (filter !== "all" && fav.item_type !== filter) return false;
      if (filters.tagIds.length > 0) {
        const ids = new Set(fav.tags.map((t) => t.id));
        if (!filters.tagIds.every((id) => ids.has(id))) return false;
      }

      const agreement = fav.agreement_uuid
        ? (agreementByUuid[fav.agreement_uuid] ?? null)
        : null;
      const ctxTarget = (
        agreement?.target ??
        (typeof fav.context?.target === "string"
          ? (fav.context.target as string)
          : "")
      )
        .toString()
        .toLowerCase();
      const ctxAcquirer = (
        agreement?.acquirer ??
        (typeof fav.context?.acquirer === "string"
          ? (fav.context.acquirer as string)
          : "")
      )
        .toString()
        .toLowerCase();

      if (targetQ && !ctxTarget.includes(targetQ)) return false;
      if (acquirerQ && !ctxAcquirer.includes(acquirerQ)) return false;

      // Year + size filters require agreement metadata to be loaded; if it
      // hasn't loaded yet, exclude when the user has set those filters.
      if (yearMin !== null || yearMax !== null) {
        const year = agreement?.year ?? null;
        if (year === null) return false;
        if (yearMin !== null && year < yearMin) return false;
        if (yearMax !== null && year > yearMax) return false;
      }
      if (sizeMin !== null || sizeMax !== null) {
        const size = agreement?.transaction_price_total ?? null;
        if (size === null) return false;
        if (sizeMin !== null && size < sizeMin) return false;
        if (sizeMax !== null && size > sizeMax) return false;
      }
      return true;
    });
  }, [activeProjectId, filter, filters, favorites, agreementByUuid]);

  useEffect(() => {
    setSelectedIds((prev) => {
      if (prev.size === 0) return prev;
      const visibleIds = new Set(visible.map((fav) => fav.id));
      const next = new Set(Array.from(prev).filter((id) => visibleIds.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [visible]);

  if (status === "loading") {
    return (
      <PageShell title="Favorites">
        <div className="text-sm text-muted-foreground">Loading…</div>
      </PageShell>
    );
  }
  if (!user) {
    return <Navigate to="/account" state={{ from: location }} replace />;
  }

  const handleUpdated = (next: Favorite) =>
    setFavorites((prev) => prev.map((f) => (f.id === next.id ? next : f)));
  const handleRemoved = (id: string) =>
    setFavorites((prev) => prev.filter((f) => f.id !== id));
  const handleTagsChanged = (id: string, tags: FavoriteTag[]) => {
    setFavoriteTagsCache(id, tags);
    setFavorites((prev) => prev.map((f) => (f.id === id ? { ...f, tags } : f)));
  };
  const handleTagUpdated = (tag: FavoriteTag) => {
    setFavorites((prev) =>
      prev.map((f) => ({
        ...f,
        tags: f.tags.map((existing) =>
          existing.id === tag.id ? tag : existing,
        ),
      })),
    );
  };
  const handleTagDeleted = (tagId: string) => {
    setFilters((prev) => ({
      ...prev,
      tagIds: prev.tagIds.filter((id) => id !== tagId),
    }));
    setFavorites((prev) =>
      prev.map((f) => ({
        ...f,
        tags: f.tags.filter((tag) => tag.id !== tagId),
      })),
    );
  };
  const handleSelectFavorite = (id: string, selected: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (selected) {
        next.add(id);
      } else {
        next.delete(id);
      }
      return next;
    });
  };
  const visibleIds = visible.map((fav) => fav.id);
  const selectedVisibleCount = visibleIds.filter((id) =>
    selectedIds.has(id),
  ).length;
  const allVisibleSelected =
    visibleIds.length > 0 && selectedVisibleCount === visibleIds.length;
  const someVisibleSelected =
    selectedVisibleCount > 0 && selectedVisibleCount < visibleIds.length;
  const handleToggleVisibleSelection = (selected: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      for (const id of visibleIds) {
        if (selected) {
          next.add(id);
        } else {
          next.delete(id);
        }
      }
      return next;
    });
  };
  const handleMoveFavorites = async (
    favoriteIds: string[],
    projectId: string,
  ) => {
    if (favoriteIds.length === 0 || !projectId) return;
    try {
      const result = await apiBulkMoveFavorites(favoriteIds, projectId);
      const moved = new Set(result.favorite_ids);
      setFavorites((prev) =>
        prev.map((fav) =>
          moved.has(fav.id)
            ? {
                ...fav,
                project_id: result.project_id,
                project_ids: [result.project_id],
              }
            : fav,
        ),
      );
      setSelectedIds((prev) => {
        const next = new Set(prev);
        for (const id of moved) next.delete(id);
        return next;
      });
      setBulkProjectId("");
    } catch {
      toast({ title: "Couldn't move favorites", variant: "destructive" });
    }
  };
  const handleCopyFavorites = async (
    favoriteIds: string[],
    projectId: string,
  ) => {
    if (favoriteIds.length === 0 || !projectId) return;
    try {
      const result = await apiBulkCopyFavorites(favoriteIds, [projectId]);
      const copied = new Set(result.favorite_ids);
      setFavorites((prev) =>
        prev.map((fav) =>
          copied.has(fav.id)
            ? {
                ...fav,
                project_ids: Array.from(
                  new Set([...fav.project_ids, ...result.project_ids]),
                ),
              }
            : fav,
        ),
      );
      setBulkCopyProjectId("");
    } catch {
      toast({ title: "Couldn't copy favorites", variant: "destructive" });
    }
  };
  const handleDragEnd = (event: DragEndEvent) => {
    const favoriteId = event.active.data.current?.favoriteId;
    const projectId = event.over?.data.current?.projectId;
    if (typeof favoriteId !== "string" || typeof projectId !== "string") return;
    const fav = favorites.find((row) => row.id === favoriteId);
    if (!fav || fav.project_id === projectId) return;
    void handleMoveFavorites([favoriteId], projectId);
  };
  const handleProjectCreated = (project: FavoriteProject) => {
    void reloadProjects();
    setActiveProjectId(project.id);
  };
  const handleProjectUpdated = (_project: FavoriteProject) => {
    void reloadProjects();
  };
  const handleProjectDeleted = (
    projectId: string,
    reassignProjectId: string,
  ) => {
    setFavorites((prev) =>
      prev.map((fav) =>
        fav.project_ids.includes(projectId) || fav.project_id === projectId
          ? {
              ...fav,
              project_id:
                fav.project_id === projectId
                  ? reassignProjectId
                  : fav.project_id,
              project_ids: Array.from(
                new Set(
                  fav.project_ids
                    .filter((id) => id !== projectId)
                    .concat(reassignProjectId),
                ),
              ),
            }
          : fav,
      ),
    );
    if (activeProjectId === projectId) {
      setActiveProjectId(reassignProjectId);
    }
    void reloadProjects();
  };

  return (
    <PageShell size="full" className="px-0 py-0 sm:px-0 lg:px-0">
      <DndContext sensors={sensors} onDragEnd={handleDragEnd}>
        <div className="w-full overflow-x-hidden">
          <div className="border-b border-border px-4 py-3 sm:px-8">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:gap-x-5">
                <h1 className="shrink-0 text-xl font-semibold tracking-tight text-foreground">
                  Favorites
                </h1>
              </div>
              <span className="text-sm text-muted-foreground">
                {visible.length} of {favorites.length} shown
              </span>
            </div>
          </div>

          <div className="border-b border-border bg-muted/20 px-4 py-2.5 backdrop-blur supports-[backdrop-filter]:bg-muted/20 sm:px-8">
            <div className="grid min-w-0 gap-2 xl:grid-cols-[18rem_minmax(0,1fr)]">
              <TagsManager
                onTagUpdated={handleTagUpdated}
                onTagDeleted={handleTagDeleted}
              />
              <FilterBar
                filters={filters}
                onChange={setFilters}
                onClear={() => setFilters(EMPTY_FILTERS)}
              />
            </div>
          </div>

          <div className="grid gap-4 px-4 py-4 sm:px-8 xl:grid-cols-[18rem_minmax(0,1fr)]">
            <ProjectSidebar
              projects={projects}
              activeProjectId={activeProjectId}
              projectCounts={projectCounts}
              onSelectProject={setActiveProjectId}
              onProjectCreated={handleProjectCreated}
              onProjectUpdated={handleProjectUpdated}
              onProjectDeleted={handleProjectDeleted}
            />

            <div className="min-w-0 space-y-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex min-w-0 flex-col gap-3 lg:flex-row lg:items-center">
                  <div className="shrink-0">
                    <div className="text-sm font-semibold text-foreground">
                      Saved items
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {visible.length} of {favorites.length}
                    </div>
                  </div>
                  <Tabs
                    value={filter}
                    onValueChange={(value) => setFilter(value as Filter)}
                    className="min-w-0"
                  >
                    <div className="max-w-full overflow-x-auto">
                      <TabsList className="min-h-9 min-w-max border bg-background px-1 shadow-sm">
                        <TabsTrigger className="px-3 text-sm" value="all">
                          All ({counts.all})
                        </TabsTrigger>
                        <TabsTrigger className="px-3 text-sm" value="section">
                          Sections ({counts.section})
                        </TabsTrigger>
                        <TabsTrigger className="px-3 text-sm" value="agreement">
                          Deals ({counts.agreement})
                        </TabsTrigger>
                        <TabsTrigger
                          className="px-3 text-sm"
                          value="tax_clause"
                        >
                          Tax clauses ({counts.tax_clause})
                        </TabsTrigger>
                      </TabsList>
                    </div>
                  </Tabs>
                </div>
                <label className="flex min-h-9 items-center gap-2 rounded-md border bg-background px-3 text-sm font-medium text-foreground shadow-sm">
                  <Checkbox
                    checked={
                      allVisibleSelected
                        ? true
                        : someVisibleSelected
                          ? "indeterminate"
                          : false
                    }
                    onCheckedChange={(checked) =>
                      handleToggleVisibleSelection(checked === true)
                    }
                    disabled={visibleIds.length === 0}
                    aria-label={
                      allVisibleSelected
                        ? "Deselect all visible favorites"
                        : "Select all visible favorites"
                    }
                  />
                  <span>
                    {allVisibleSelected ? "Deselect all" : "Select all"}
                  </span>
                </label>
              </div>

              {selectedIds.size > 0 ? (
                <div className="sticky top-16 z-20 flex flex-col gap-2 rounded-lg border bg-background/95 px-3 py-2 shadow-sm backdrop-blur sm:flex-row sm:items-center">
                  <span className="text-sm font-medium text-foreground">
                    {selectedIds.size} selected
                  </span>
                  <div className="flex flex-wrap items-center gap-2">
                    <Select
                      value={bulkProjectId}
                      onValueChange={setBulkProjectId}
                    >
                      <SelectTrigger
                        className="h-11 w-full text-sm sm:h-8 sm:w-44"
                        aria-label="Move selected favorites to project"
                      >
                        <SelectValue placeholder="Move to project" />
                      </SelectTrigger>
                      <SelectContent>
                        {projects.map((project) => (
                          <SelectItem key={project.id} value={project.id}>
                            {project.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button
                      type="button"
                      size="sm"
                      className="h-11 sm:h-8"
                      disabled={!bulkProjectId}
                      onClick={() =>
                        void handleMoveFavorites(
                          Array.from(selectedIds),
                          bulkProjectId,
                        )
                      }
                    >
                      Move
                    </Button>
                    <Select
                      value={bulkCopyProjectId}
                      onValueChange={setBulkCopyProjectId}
                    >
                      <SelectTrigger
                        className="h-11 w-full text-sm sm:h-8 sm:w-44"
                        aria-label="Copy selected favorites to project"
                      >
                        <SelectValue placeholder="Copy to project" />
                      </SelectTrigger>
                      <SelectContent>
                        {projects.map((project) => (
                          <SelectItem key={project.id} value={project.id}>
                            {project.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="h-11 sm:h-8"
                      disabled={!bulkCopyProjectId}
                      onClick={() =>
                        void handleCopyFavorites(
                          Array.from(selectedIds),
                          bulkCopyProjectId,
                        )
                      }
                    >
                      Copy
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-11 sm:h-8"
                      onClick={() => setSelectedIds(new Set())}
                    >
                      Clear
                    </Button>
                  </div>
                </div>
              ) : null}

              {loading ? (
                <div className="text-sm text-muted-foreground">Loading…</div>
              ) : visible.length === 0 ? (
                <Card className="flex flex-col items-center gap-2 p-8 text-center">
                  <Star className="h-6 w-6 text-muted-foreground" />
                  <div className="text-sm font-medium">
                    {favorites.length === 0
                      ? "No favorites yet"
                      : "No favorites match these filters"}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {favorites.length === 0
                      ? "Click the star on any search result or agreement to save it here."
                      : "Try clearing some filters."}
                  </div>
                </Card>
              ) : (
                <div className="space-y-2">
                  {visible.map((fav) => (
                    <FavoriteRow
                      key={fav.id}
                      fav={fav}
                      agreement={
                        fav.agreement_uuid
                          ? (agreementByUuid[fav.agreement_uuid] ?? null)
                          : null
                      }
                      sectionDetails={
                        fav.item_type === "section"
                          ? (sectionByUuid[fav.item_uuid] ?? null)
                          : null
                      }
                      clauseTypeLabelById={clauseTypeLabelById}
                      selected={selectedIds.has(fav.id)}
                      onSelectChange={handleSelectFavorite}
                      onUpdated={handleUpdated}
                      onRemoved={handleRemoved}
                      onTagsChanged={handleTagsChanged}
                      onTagDeleted={handleTagDeleted}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </DndContext>
    </PageShell>
  );
}
