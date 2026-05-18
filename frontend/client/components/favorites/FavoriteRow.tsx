import { CSS } from "@dnd-kit/utilities";
import { useDraggable } from "@dnd-kit/core";
import { useCallback, useState } from "react";
import { Link } from "react-router-dom";
import { ExternalLink, GripVertical, Trash2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import { TagEditor } from "@/components/favorites/TagEditor";
import { TagPill } from "@/components/favorites/TagPill";
import { useToast } from "@/hooks/use-toast";
import {
  deleteFavorite as apiDeleteFavorite,
  patchFavorite as apiPatchFavorite,
  setFavoriteTags as apiSetFavoriteTags,
  type Favorite,
  type FavoriteTag,
} from "@/lib/favorites-api";
import { formatCompactCurrencyValue } from "@/lib/format-utils";
import type { Agreement } from "@shared/agreement";

import {
  contextString,
  favoriteHeading,
  favoriteHref,
  firstWords,
  formatDate,
  stripXmlText,
} from "./helpers";
import { TYPE_LABELS, type SectionDetails } from "./types";

export function FavoriteRow({
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
