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
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { useFavorites } from "@/contexts/FavoritesContext";
import { TagEditor } from "@/components/favorites/TagEditor";
import { TagPill } from "@/components/favorites/TagPill";
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
    tags,
    tagsForFavorite,
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
  const [optimisticStarred, setOptimisticStarred] = useState<boolean | null>(
    null,
  );
  const clickTimerRef = useRef<number | null>(null);
  const pendingClickRef = useRef<boolean>(false);
  const ignoreSingleRef = useRef<boolean>(false);

  useEffect(() => {
    if (user && itemUuid) {
      ensureLoaded(itemType, [itemUuid]);
    }
  }, [user, itemType, itemUuid, ensureLoaded]);

  useEffect(
    () => () => {
      if (clickTimerRef.current !== null) {
        window.clearTimeout(clickTimerRef.current);
      }
    },
    [],
  );

  const starred = user ? isStarred(itemType, itemUuid) : false;
  const displayedStarred = optimisticStarred ?? starred;

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
        const id = favoriteIdFor(itemType, itemUuid);
        if (id) {
          await deleteFavorite(itemType, itemUuid, id);
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
    favoriteIdFor,
    deleteFavorite,
    upsertFavorite,
    baseContext,
    toast,
  ]);

  const handleClick = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();
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

  const handleDoubleClick = useCallback(
    async (event: React.MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();
      if (!user) {
        requireSignIn();
        return;
      }
      // Cancel the pending single-click toggle.
      if (clickTimerRef.current !== null) {
        window.clearTimeout(clickTimerRef.current);
        clickTimerRef.current = null;
      }
      pendingClickRef.current = false;
      ignoreSingleRef.current = true;

      // Ensure favorited (idempotent on backend); preload existing note.
      setNote("");
      setDraftTagIds([]);
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
    },
    [
      user,
      starred,
      displayedStarred,
      itemType,
      itemUuid,
      upsertFavorite,
      favoriteIdFor,
      tagsForFavorite,
      baseContext,
      toast,
    ],
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
      await patchFavorite(favoriteId, { note: note.trim() || null });
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

  const iconSize = size === "md" ? 18 : 16;
  const button = (
    <button
      type="button"
      className={cn(
        "inline-flex h-8 w-8 select-none items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        displayedStarred && "text-amber-500 hover:text-amber-500",
        className,
      )}
      aria-label={
        ariaLabel ??
        (displayedStarred
          ? "Unfavorite (double-click to add note)"
          : "Favorite")
      }
      aria-pressed={displayedStarred}
      title={displayedStarred ? "Starred — double-click to add a note" : "Star"}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
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
    <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
      <PopoverTrigger asChild>{button}</PopoverTrigger>
      <PopoverContent
        align="end"
        className="w-72 space-y-2"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="text-sm font-medium">Add a note</div>
        <Textarea
          autoFocus
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
          <div className="text-xs font-medium text-muted-foreground">Tags</div>
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
                    setDraftTagIds((prev) => prev.filter((id) => id !== tag.id))
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
  );
}
