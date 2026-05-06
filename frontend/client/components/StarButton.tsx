import { useCallback, useEffect, useRef, useState } from "react";
import { Star } from "lucide-react";
import { useLocation } from "react-router-dom";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { useFavorites } from "@/contexts/FavoritesContext";
import { TagEditor } from "@/components/favorites/TagEditor";
import { TagPill, TagSwatch } from "@/components/favorites/TagPill";
import { cn } from "@/lib/utils";
import type { FavoriteItemType } from "@/lib/favorites-api";

interface StarButtonProps {
  itemType: FavoriteItemType;
  itemUuid: string;
  /** Extra context captured at star-time (search params, position, etc.). */
  context?: Record<string, unknown>;
  /** Override label for accessibility. */
  ariaLabel?: string;
  className?: string;
  size?: "sm" | "md";
}

const DOUBLE_CLICK_MS = 300;
const TOUCH_LONG_PRESS_MS = 550;

export function StarButton({
  itemType,
  itemUuid,
  context,
  ariaLabel,
  className,
  size = "sm",
}: StarButtonProps) {
  const { user } = useAuth();
  const { toast } = useToast();
  const location = useLocation();
  const {
    isStarred,
    favoriteIdFor,
    upsertFavorite,
    deleteFavorite,
    patchFavorite,
    ensureLoaded,
    projects,
    ensureProjectsLoaded,
    tags,
    tagsForFavorite,
    loadTagsForFavorite,
    setFavoriteTags,
  } = useFavorites();

  const [popoverOpen, setPopoverOpen] = useState(false);
  const [popoverFavoriteId, setPopoverFavoriteId] = useState<string | null>(
    null,
  );
  const [note, setNote] = useState("");
  const [savingNote, setSavingNote] = useState(false);
  const [openingFavorite, setOpeningFavorite] = useState(false);
  const [draftTagIds, setDraftTagIds] = useState<string[]>([]);
  const [draftProjectId, setDraftProjectId] = useState<string | null>(null);
  const [optimisticStarred, setOptimisticStarred] = useState<boolean | null>(
    null,
  );
  const clickTimerRef = useRef<number | null>(null);
  const longPressTimerRef = useRef<number | null>(null);
  const pendingClickRef = useRef<boolean>(false);
  const ignoreSingleRef = useRef<boolean>(false);
  const suppressNextClickRef = useRef<boolean>(false);

  useEffect(() => {
    if (user && itemUuid) {
      ensureLoaded(itemType, [itemUuid]);
    }
  }, [user, itemType, itemUuid, ensureLoaded]);

  useEffect(() => {
    if (popoverOpen && user) {
      ensureProjectsLoaded();
    }
  }, [popoverOpen, user, ensureProjectsLoaded]);

  useEffect(
    () => () => {
      if (clickTimerRef.current !== null) {
        window.clearTimeout(clickTimerRef.current);
      }
      if (longPressTimerRef.current !== null) {
        window.clearTimeout(longPressTimerRef.current);
      }
    },
    [],
  );

  const starred = user ? isStarred(itemType, itemUuid) : false;
  const displayedStarred = optimisticStarred ?? starred;
  const favoriteId = favoriteIdFor(itemType, itemUuid);
  const displayTags = favoriteId ? (tagsForFavorite(favoriteId) ?? []) : [];

  useEffect(() => {
    if (!displayedStarred || !favoriteId || tagsForFavorite(favoriteId)) return;
    void loadTagsForFavorite(favoriteId).catch(() => {
      // Tags are decorative in the star affordance; the editor can retry.
    });
  }, [displayedStarred, favoriteId, tagsForFavorite, loadTagsForFavorite]);

  useEffect(() => {
    if (optimisticStarred !== null && optimisticStarred === starred) {
      setOptimisticStarred(null);
    }
  }, [optimisticStarred, starred]);

  const baseContext = useCallback(
    () => ({
      source_path: location.pathname + location.search,
      ...(context ?? {}),
      starred_at: new Date().toISOString(),
    }),
    [context, location.pathname, location.search],
  );

  const requireSignIn = () => {
    toast({
      title: "Sign in to favorite",
      description: "Sign in or create an account to star results.",
    });
  };

  const performToggle = useCallback(async () => {
    if (!user) {
      requireSignIn();
      return;
    }
    try {
      if (displayedStarred) {
        setOptimisticStarred(false);
        if (favoriteId) {
          await deleteFavorite(itemType, itemUuid, favoriteId);
        }
      } else {
        setOptimisticStarred(true);
        await upsertFavorite({
          item_type: itemType,
          item_uuid: itemUuid,
          context: baseContext(),
        });
      }
    } catch {
      setOptimisticStarred(starred);
      toast({
        title: "Couldn't update favorite",
        description: "Please try again.",
        variant: "destructive",
      });
    }
  }, [
    user,
    starred,
    displayedStarred,
    itemType,
    itemUuid,
    favoriteId,
    deleteFavorite,
    upsertFavorite,
    baseContext,
    toast,
  ]);

  const handleClick = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();
      if (suppressNextClickRef.current) {
        suppressNextClickRef.current = false;
        return;
      }
      if (!user) {
        requireSignIn();
        return;
      }
      if (event.detail > 1) {
        if (clickTimerRef.current !== null) {
          window.clearTimeout(clickTimerRef.current);
          clickTimerRef.current = null;
        }
        pendingClickRef.current = false;
        return;
      }
      if (ignoreSingleRef.current) {
        // The second click of a double-click landed; popover already opened.
        ignoreSingleRef.current = false;
        return;
      }
      if (pendingClickRef.current) return;
      pendingClickRef.current = true;
      clickTimerRef.current = window.setTimeout(() => {
        pendingClickRef.current = false;
        clickTimerRef.current = null;
        void performToggle();
      }, DOUBLE_CLICK_MS);
    },
    [performToggle, user],
  );

  const openFavoriteEditor = useCallback(async () => {
    if (!user) {
      requireSignIn();
      return;
    }

    // Ensure favorited (idempotent on backend); preload existing note.
    setNote("");
    setDraftTagIds([]);
    setDraftProjectId(null);
    setPopoverFavoriteId(favoriteIdFor(itemType, itemUuid));
    setPopoverOpen(true);
    try {
      let id = favoriteIdFor(itemType, itemUuid);
      if (!displayedStarred) {
        setOptimisticStarred(true);
        setOpeningFavorite(true);
        const result = await upsertFavorite({
          item_type: itemType,
          item_uuid: itemUuid,
          context: baseContext(),
        });
        id = result.id;
        setDraftProjectId(result.project_id);
      }
      const existing = id ? tagsForFavorite(id) : undefined;
      setPopoverFavoriteId(id);
      setDraftTagIds(existing ? existing.map((t) => t.id) : []);
    } catch {
      setPopoverOpen(false);
      toast({
        title: "Couldn't open favorite",
        description: "Please try again.",
        variant: "destructive",
      });
    } finally {
      setOpeningFavorite(false);
    }
  }, [
    user,
    displayedStarred,
    itemType,
    itemUuid,
    upsertFavorite,
    favoriteIdFor,
    tagsForFavorite,
    baseContext,
    toast,
  ]);

  const handleDoubleClick = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();
      // Cancel the pending single-click toggle.
      if (clickTimerRef.current !== null) {
        window.clearTimeout(clickTimerRef.current);
        clickTimerRef.current = null;
      }
      pendingClickRef.current = false;
      ignoreSingleRef.current = true;
      void openFavoriteEditor();
    },
    [openFavoriteEditor],
  );

  const cancelLongPress = useCallback(() => {
    if (longPressTimerRef.current !== null) {
      window.clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
  }, []);

  const handlePointerDown = useCallback(
    (event: React.PointerEvent) => {
      if (event.pointerType !== "touch") return;
      event.stopPropagation();
      cancelLongPress();
      longPressTimerRef.current = window.setTimeout(() => {
        longPressTimerRef.current = null;
        suppressNextClickRef.current = true;
        if (clickTimerRef.current !== null) {
          window.clearTimeout(clickTimerRef.current);
          clickTimerRef.current = null;
        }
        pendingClickRef.current = false;
        void openFavoriteEditor();
      }, TOUCH_LONG_PRESS_MS);
    },
    [cancelLongPress, openFavoriteEditor],
  );

  const handlePointerEnd = useCallback(() => {
    cancelLongPress();
  }, [cancelLongPress]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (
        (event.key === "Enter" && event.shiftKey) ||
        event.key === "ArrowDown"
      ) {
        event.preventDefault();
        event.stopPropagation();
        if (clickTimerRef.current !== null) {
          window.clearTimeout(clickTimerRef.current);
          clickTimerRef.current = null;
        }
        pendingClickRef.current = false;
        void openFavoriteEditor();
      }
    },
    [openFavoriteEditor],
  );

  const handleSaveNote = useCallback(async () => {
    if (!user) return;
    const id = favoriteIdFor(itemType, itemUuid);
    const favoriteId = popoverFavoriteId ?? id;
    if (!favoriteId) return;
    setSavingNote(true);
    try {
      const existing = tagsForFavorite(favoriteId) ?? [];
      const existingIds = existing.map((t) => t.id).sort();
      const draftSorted = [...draftTagIds].sort();
      const tagsChanged =
        existingIds.length !== draftSorted.length ||
        existingIds.some((v, i) => v !== draftSorted[i]);
      await patchFavorite(favoriteId, {
        note: note.trim() || null,
        ...(draftProjectId ? { project_id: draftProjectId } : {}),
      });
      if (tagsChanged) {
        await setFavoriteTags(favoriteId, draftTagIds);
      }
      setPopoverOpen(false);
      toast({ title: "Saved" });
    } catch {
      toast({
        title: "Couldn't save",
        description: "Please try again.",
        variant: "destructive",
      });
    } finally {
      setSavingNote(false);
    }
  }, [
    user,
    favoriteIdFor,
    popoverFavoriteId,
    tagsForFavorite,
    draftTagIds,
    draftProjectId,
    itemType,
    itemUuid,
    patchFavorite,
    setFavoriteTags,
    note,
    toast,
  ]);

  const draftTags = tags.filter((tag) => draftTagIds.includes(tag.id));
  const canSaveNote =
    !savingNote &&
    !openingFavorite &&
    Boolean(popoverFavoriteId ?? favoriteIdFor(itemType, itemUuid));

  const handleDraftTagsChange = useCallback((nextTagIds: string[]) => {
    setDraftTagIds(nextTagIds);
  }, []);

  const iconSize = size === "md" ? 20 : 18;
  const defaultAriaLabel = displayedStarred
    ? "Remove from favorites. Press Shift Enter, Arrow Down, or touch and hold to edit favorite details."
    : "Add to favorites. Press Shift Enter, Arrow Down, or touch and hold to add favorite details.";
  const title = displayedStarred
    ? "Remove from favorites; Shift+Enter, Arrow Down, or touch and hold to edit details"
    : "Add to favorites; Shift+Enter, Arrow Down, or touch and hold to add details";
  const button = (
    <button
      type="button"
      className={cn(
        "inline-flex h-11 w-11 select-none items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        displayedStarred && "text-amber-500 hover:text-amber-500",
      )}
      aria-label={ariaLabel ?? defaultAriaLabel}
      aria-pressed={displayedStarred}
      title={title}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerEnd}
      onPointerCancel={handlePointerEnd}
      onPointerLeave={handlePointerEnd}
      onKeyDown={handleKeyDown}
    >
      <Star
        size={iconSize}
        className={cn(displayedStarred ? "fill-current" : "fill-none")}
        strokeWidth={2}
        aria-hidden="true"
      />
    </button>
  );

  return (
    <div className={cn("inline-flex flex-col items-center", className)}>
      <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
        <PopoverTrigger asChild>{button}</PopoverTrigger>
        <PopoverContent
          align="end"
          className="w-[min(22rem,calc(100vw-2rem))] space-y-2"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="text-sm font-medium">Add a note</div>
          <Textarea
            autoFocus
            aria-label="Favorite note"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (canSaveNote) void handleSaveNote();
              }
            }}
            placeholder="Why is this interesting? Anything to remember… (Enter to save, Shift+Enter for newline)"
            rows={4}
          />
          <div className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground">
              Tags
            </div>
            <TagEditor
              selectedTagIds={draftTagIds}
              onChange={handleDraftTagsChange}
              showSelectedTags={false}
              triggerLabel={draftTagIds.length === 0 ? "Add tags" : "Edit tags"}
              helperText="Select tags, then save."
            />
            <div className="flex min-h-7 flex-wrap items-center gap-1.5 pt-1">
              {draftTags.length > 0 ? (
                draftTags.map((tag) => (
                  <TagPill
                    key={tag.id}
                    name={tag.name}
                    color={tag.color}
                    onRemove={() =>
                      setDraftTagIds((prev) =>
                        prev.filter((id) => id !== tag.id),
                      )
                    }
                  />
                ))
              ) : (
                <span className="text-xs text-muted-foreground">
                  No tags selected.
                </span>
              )}
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground">
              Project
            </div>
            <Select
              value={draftProjectId ?? ""}
              onValueChange={(value) => setDraftProjectId(value || null)}
            >
              <SelectTrigger
                className="h-11 text-sm"
                aria-label="Favorite project"
              >
                <SelectValue placeholder="Leave unchanged" />
              </SelectTrigger>
              <SelectContent>
                {projects.map((project) => (
                  <SelectItem key={project.id} value={project.id}>
                    {project.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setPopoverOpen(false)}
              disabled={savingNote}
            >
              Cancel
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={() => void handleSaveNote()}
              disabled={!canSaveNote}
            >
              {savingNote ? "Saving…" : openingFavorite ? "Opening…" : "Save"}
            </Button>
          </div>
        </PopoverContent>
      </Popover>
      {displayTags.length > 0 ? (
        <div
          className="mt-0.5 flex max-w-12 flex-wrap justify-center gap-0.5"
          aria-label={`Tags: ${displayTags.map((tag) => tag.name).join(", ")}`}
        >
          {displayTags.slice(0, 6).map((tag) => (
            <span key={tag.id} title={tag.name}>
              <TagSwatch color={tag.color} className="h-2.5 w-2.5" />
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
