import { useEffect, useState } from "react";
import { Check, ChevronDown, Pencil, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { TagPill, TagSwatch } from "@/components/favorites/TagPill";
import { useFavorites } from "@/contexts/FavoritesContext";
import { useToast } from "@/hooks/use-toast";
import {
  TAG_COLORS,
  type FavoriteTag,
  type TagColor,
} from "@/lib/favorites-api";

export function TagsManager({
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
