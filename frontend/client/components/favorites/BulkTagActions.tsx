import { useEffect, useState } from "react";
import { Check, Plus, Tag as TagIcon } from "lucide-react";

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
import { TAG_COLORS, type TagColor } from "@/lib/favorites-api";

export function BulkTagActions({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (tagIds: string[], action: "add" | "remove") => Promise<void>;
}) {
  const { toast } = useToast();
  const { tags, ensureTagsLoaded, createTag } = useFavorites();
  const [open, setOpen] = useState(false);
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState<TagColor>("blue");
  const [busy, setBusy] = useState(false);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    ensureTagsLoaded();
  }, [ensureTagsLoaded]);

  const toggleTag = (tagId: string) => {
    setSelectedTagIds((prev) =>
      prev.includes(tagId)
        ? prev.filter((id) => id !== tagId)
        : [...prev, tagId],
    );
  };

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      const tag = await createTag(name, newColor);
      setSelectedTagIds((prev) =>
        prev.includes(tag.id) ? prev : [...prev, tag.id],
      );
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

  const handleSubmit = async (action: "add" | "remove") => {
    if (selectedTagIds.length === 0) return;
    setBusy(true);
    try {
      await onSubmit(selectedTagIds, action);
      setSelectedTagIds([]);
      setOpen(false);
    } catch {
      // The caller owns the user-facing error toast.
    } finally {
      setBusy(false);
    }
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="h-11 gap-1.5 sm:h-8"
          disabled={disabled}
        >
          <TagIcon className="h-3.5 w-3.5" aria-hidden="true" />
          Apply/remove tags
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={8}
        className="w-[min(20rem,calc(100vw-2rem))] space-y-3"
      >
        <div className="space-y-2">
          <div className="text-xs font-medium text-muted-foreground">
            Bulk tags
          </div>
          {tags.length > 0 ? (
            <div className="max-h-40 space-y-1 overflow-y-auto pr-1">
              {tags.map((tag) => (
                <TagPill
                  key={tag.id}
                  name={tag.name}
                  color={tag.color}
                  selected={selectedTagIds.includes(tag.id)}
                  onClick={() => toggleTag(tag.id)}
                  className="max-w-full"
                />
              ))}
            </div>
          ) : (
            <div className="text-xs text-muted-foreground">No tags yet.</div>
          )}
        </div>

        <div className="space-y-2 border-t pt-2">
          <div className="text-xs font-medium">New tag</div>
          <Input
            aria-label="New bulk tag name"
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

        <div className="flex justify-end gap-2 border-t pt-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-11 sm:h-8"
            disabled={busy || selectedTagIds.length === 0}
            onClick={() => void handleSubmit("remove")}
          >
            Remove
          </Button>
          <Button
            type="button"
            size="sm"
            className="h-11 sm:h-8"
            disabled={busy || selectedTagIds.length === 0}
            onClick={() => void handleSubmit("add")}
          >
            Apply
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
