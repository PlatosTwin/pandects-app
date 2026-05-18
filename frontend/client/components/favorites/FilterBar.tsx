import { useEffect, useState } from "react";
import { ChevronDown, Filter, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
import { TagPill } from "@/components/favorites/TagPill";
import { useFavorites } from "@/contexts/FavoritesContext";

import { hasActiveFilters } from "./helpers";
import {
  ANY_SIZE_VALUE,
  SIZE_OPTIONS,
  type FavoriteFilters,
} from "./types";

export function FilterBar({
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
