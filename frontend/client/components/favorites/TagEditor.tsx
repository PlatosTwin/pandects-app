import { useEffect, useMemo, useState } from "react";
import { Check, Plus, Tag as TagIcon, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useToast } from "@/hooks/use-toast";
import { useFavorites } from "@/contexts/FavoritesContext";
import {
  TAG_COLORS,
  type FavoriteTag,
  type TagColor,
} from "@/lib/favorites-api";
import { TagPill, TagSwatch } from "./TagPill";

interface TagEditorProps {
  selectedTagIds: string[];
  onChange: (tagIds: string[]) => void;
  onTagDeleted?: (tagId: string) => void;
  showSelectedTags?: boolean;
  triggerLabel?: string;
  helperText?: string;
}

export function TagEditor({
  selectedTagIds,
  onChange,
  onTagDeleted,
  showSelectedTags = true,
  triggerLabel,
  helperText = "Select a tag to attach it now.",
}: TagEditorProps) {
  const { toast } = useToast();
  const { tags, ensureTagsLoaded, createTag, removeTag } = useFavorites();
  const [open, setOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState<TagColor>("blue");
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    ensureTagsLoaded();
  }, [ensureTagsLoaded]);

  const selectedTags = useMemo(
    () => tags.filter((t) => selectedTagIds.includes(t.id)),
    [tags, selectedTagIds],
  );

  const toggle = (tagId: string) => {
    if (selectedTagIds.includes(tagId)) {
      onChange(selectedTagIds.filter((id) => id !== tagId));
    } else {
      onChange([...selectedTagIds, tagId]);
    }
  };

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      const tag = await createTag(name, newColor);
      if (!selectedTagIds.includes(tag.id)) {
        onChange([...selectedTagIds, tag.id]);
      }
      setNewName("");
    } catch {
      toast({
        title: "Couldn't create tag",
        description: "Please try again.",
        variant: "destructive",
      });
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteTag = async (tag: FavoriteTag) => {
    const confirmed = window.confirm(
      `Delete "${tag.name}" permanently? This removes it from every favorite.`,
    );
    if (!confirmed) return;
    try {
      await removeTag(tag.id);
      if (selectedTagIds.includes(tag.id)) {
        onChange(selectedTagIds.filter((id) => id !== tag.id));
      }
      onTagDeleted?.(tag.id);
    } catch {
      toast({
        title: "Couldn't delete tag",
        description: "Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {showSelectedTags
        ? selectedTags.map((t: FavoriteTag) => (
            <TagPill
              key={t.id}
              name={t.name}
              color={t.color}
              onRemove={() => toggle(t.id)}
            />
          ))
        : null}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-11 gap-1 px-3 text-xs sm:h-6 sm:px-2"
          >
            <TagIcon className="h-3 w-3" aria-hidden="true" />
            {triggerLabel ??
              (selectedTags.length === 0 ? "Add tag" : "Edit tags")}
          </Button>
        </PopoverTrigger>
        <PopoverContent
          className="w-[min(18rem,calc(100vw-2rem))] space-y-2"
          onClick={(e) => e.stopPropagation()}
        >
          {tags.length > 0 ? (
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-2">
                <div className="text-[11px] text-muted-foreground">
                  {helperText}
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-11 px-3 text-xs sm:h-6 sm:px-2"
                  onClick={() => setOpen(false)}
                >
                  Done
                </Button>
              </div>
              <div className="max-h-36 space-y-1 overflow-y-auto pr-1">
                {tags.map((t) => (
                  <div
                    key={t.id}
                    className="grid grid-cols-[2.75rem_auto] items-center gap-1.5 sm:grid-cols-[1.5rem_auto]"
                  >
                    <button
                      type="button"
                      className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-destructive/10 hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:h-6 sm:w-6"
                      aria-label={`Delete tag ${t.name}`}
                      title={`Delete tag ${t.name}`}
                      onClick={() => void handleDeleteTag(t)}
                    >
                      <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                    </button>
                    <TagPill
                      name={t.name}
                      color={t.color}
                      selected={selectedTagIds.includes(t.id)}
                      onClick={() => toggle(t.id)}
                      className="max-w-[13rem] justify-self-start"
                    />
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-xs text-muted-foreground">No tags yet.</div>
          )}
          <div className="space-y-2 border-t pt-2">
            <div className="text-xs font-medium">New tag</div>
            <Input
              aria-label="New tag name"
              placeholder="Tag name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  if (!creating) void handleCreate();
                }
              }}
              className="h-11 text-sm sm:h-8"
            />
            <div className="flex flex-wrap items-center gap-1.5">
              {TAG_COLORS.map((color) => (
                <button
                  key={color}
                  type="button"
                  onClick={() => setNewColor(color)}
                  aria-label={`Use ${color} color`}
                  aria-pressed={newColor === color}
                  className={
                    "relative inline-grid h-11 w-11 place-items-center rounded-full ring-1 ring-transparent transition-shadow sm:h-5 sm:w-5 " +
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
              <Button
                type="button"
                size="sm"
                className="ml-auto h-11 gap-1 px-3 text-xs sm:h-7 sm:px-2"
                onClick={() => void handleCreate()}
                disabled={creating || !newName.trim()}
              >
                <Plus className="h-3 w-3" />
                Create
              </Button>
            </div>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
