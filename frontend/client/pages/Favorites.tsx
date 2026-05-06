import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, Navigate, useLocation } from "react-router-dom";
import {
  Check,
  ChevronDown,
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
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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
import {
  deleteFavorite as apiDeleteFavorite,
  listFavorites,
  patchFavorite as apiPatchFavorite,
  setFavoriteTags as apiSetFavoriteTags,
  TAG_COLORS,
  type Favorite,
  type FavoriteItemType,
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

function formatDate(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString();
}

function FavoriteRow({
  fav,
  agreement,
  onUpdated,
  onRemoved,
  onTagsChanged,
  onTagDeleted,
}: {
  fav: Favorite;
  agreement: Agreement | null;
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

  return (
    <Card className="relative overflow-hidden border-l-4 border-l-amber-400 bg-card shadow-sm transition-shadow hover:shadow-md">
      <div className="flex flex-col gap-3 p-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex flex-wrap items-center gap-1.5">
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
                ${(dealValue / 1_000_000).toFixed(1)}M
              </span>
            ) : null}
          </div>
          <div className="text-base font-semibold leading-snug text-foreground">
            {heading}
          </div>

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
              className="block max-w-full whitespace-pre-wrap break-words text-left text-sm text-muted-foreground hover:text-foreground"
            >
              {fav.note ?? "Add a note…"}
            </button>
          )}
        </div>
        <div className="flex shrink-0 gap-2">
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
  const [open, setOpen] = useState(true);
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
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card className="border-primary/20 bg-background p-3 shadow-sm">
        <div className="grid grid-cols-[1fr_auto] items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <Filter className="h-3.5 w-3.5" aria-hidden="true" />
            Filters
            {active ? (
              <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-foreground">
                Active
              </span>
            ) : null}
          </div>
          <div className="flex min-w-[8.5rem] justify-end gap-1">
            <Button
              variant="ghost"
              size="sm"
              className={`h-7 gap-1 px-2 text-xs ${active ? "" : "invisible"}`}
              onClick={onClear}
              aria-hidden={!active}
              tabIndex={active ? 0 : -1}
            >
              <X className="h-3 w-3" /> Clear
            </Button>
            <CollapsibleTrigger asChild>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-7 gap-1 px-2 text-xs"
                aria-label={open ? "Collapse filters" : "Expand filters"}
              >
                <ChevronDown
                  className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
                  aria-hidden="true"
                />
                {open ? "Hide" : "Show"}
              </Button>
            </CollapsibleTrigger>
          </div>
        </div>
        <CollapsibleContent className="mt-3 space-y-3">
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
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            <label className="space-y-1 text-xs">
              <div className="text-muted-foreground">Target</div>
              <Input
                value={filters.target}
                onChange={(e) =>
                  onChange({ ...filters, target: e.target.value })
                }
                placeholder="e.g. Acme"
                className="h-8 text-sm"
              />
            </label>
            <label className="space-y-1 text-xs">
              <div className="text-muted-foreground">Acquirer</div>
              <Input
                value={filters.acquirer}
                onChange={(e) =>
                  onChange({ ...filters, acquirer: e.target.value })
                }
                placeholder="e.g. Globex"
                className="h-8 text-sm"
              />
            </label>
            <div className="space-y-1 text-xs">
              <div className="text-muted-foreground">Year</div>
              <div className="flex items-center gap-1">
                <Input
                  type="number"
                  value={filters.yearMin}
                  onChange={(e) =>
                    onChange({ ...filters, yearMin: e.target.value })
                  }
                  placeholder="From"
                  className="h-8 text-sm"
                />
                <span aria-hidden="true">-</span>
                <Input
                  type="number"
                  value={filters.yearMax}
                  onChange={(e) =>
                    onChange({ ...filters, yearMax: e.target.value })
                  }
                  placeholder="To"
                  className="h-8 text-sm"
                />
              </div>
            </div>
            <div className="space-y-1 text-xs lg:col-span-2">
              <div className="text-muted-foreground">
                Transaction size (USD)
              </div>
              <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-1">
                <Select
                  value={filters.sizeMinUsd || ANY_SIZE_VALUE}
                  onValueChange={(value) => setSize("sizeMinUsd", value)}
                >
                  <SelectTrigger className="h-8 text-sm">
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
                  <SelectTrigger className="h-8 text-sm">
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
        </CollapsibleContent>
      </Card>
    </Collapsible>
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
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card className="border-primary/20 bg-background p-3 shadow-sm">
        <div className="grid grid-cols-[1fr_auto] items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Tags
            <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-foreground">
              {tags.length}
            </span>
          </div>
          <CollapsibleTrigger asChild>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-7 gap-1 px-2 text-xs"
              aria-label={open ? "Collapse tag manager" : "Expand tag manager"}
            >
              <ChevronDown
                className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
                aria-hidden="true"
              />
              {open ? "Hide" : "Manage"}
            </Button>
          </CollapsibleTrigger>
        </div>
        <CollapsibleContent className="mt-3">
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
                          value={draftName}
                          onChange={(e) => setDraftName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !busy) {
                              e.preventDefault();
                              void handleSave(tag);
                            }
                          }}
                          className="h-8 text-sm"
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
                                "relative inline-grid h-6 w-6 place-items-center rounded-full ring-1 ring-transparent transition-shadow " +
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
                            className="h-7 px-2 text-xs"
                            onClick={cancelEdit}
                            disabled={busy}
                          >
                            Cancel
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            className="h-7 px-2 text-xs"
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
                            className="h-7 w-7"
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
                            className="h-7 w-7 text-muted-foreground hover:text-destructive"
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
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}

function fetchAgreementMetadata(
  agreementUuid: string,
): Promise<Agreement | null> {
  return authFetch(apiUrl(`v1/agreements/${agreementUuid}`))
    .then((r) => (r.ok ? (r.json() as Promise<Agreement>) : null))
    .catch(() => null);
}

export default function FavoritesPage() {
  const { status, user } = useAuth();
  const location = useLocation();
  const { toast } = useToast();
  const { setFavoriteTagsCache } = useFavorites();
  const [filter, setFilter] = useState<Filter>("all");
  const [favorites, setFavorites] = useState<Favorite[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<FavoriteFilters>(EMPTY_FILTERS);
  const [agreementByUuid, setAgreementByUuid] = useState<
    Record<string, Agreement | null>
  >({});

  useEffect(() => {
    if (status !== "authenticated" || !user) return;
    setLoading(true);
    listFavorites()
      .then((rows) => {
        setFavorites(rows);
        for (const fav of rows) {
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
  }, [status, user, toast, setFavoriteTagsCache]);

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
  }, [filter, filters, favorites, agreementByUuid]);

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

  return (
    <PageShell
      title="Favorites"
      subtitle="Sections, deals, and tax clauses you've starred."
    >
      <div className="space-y-4">
        <Tabs value={filter} onValueChange={(v) => setFilter(v as Filter)}>
          <TabsList>
            <TabsTrigger value="all">All ({counts.all})</TabsTrigger>
            <TabsTrigger value="section">
              Sections ({counts.section})
            </TabsTrigger>
            <TabsTrigger value="agreement">
              Deals ({counts.agreement})
            </TabsTrigger>
            <TabsTrigger value="tax_clause">
              Tax clauses ({counts.tax_clause})
            </TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 shadow-sm">
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-foreground">
                  Organize favorites
                </div>
                <div className="text-xs text-muted-foreground">
                  Manage tags and narrow saved items before reviewing results.
                </div>
              </div>
              <Badge variant="secondary" className="shrink-0">
                {visible.length} shown
              </Badge>
            </div>
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

        <div className="flex items-center justify-between border-t pt-4">
          <div className="text-sm font-semibold text-foreground">
            Saved items
          </div>
          <div className="text-xs text-muted-foreground">
            {visible.length} of {favorites.length}
          </div>
        </div>

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
                onUpdated={handleUpdated}
                onRemoved={handleRemoved}
                onTagsChanged={handleTagsChanged}
                onTagDeleted={handleTagDeleted}
              />
            ))}
          </div>
        )}
      </div>
    </PageShell>
  );
}
