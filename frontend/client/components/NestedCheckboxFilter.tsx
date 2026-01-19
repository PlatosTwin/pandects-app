import { useEffect, useMemo, useRef, useState, useId } from "react";
import { ChevronDown, ChevronRight, Check, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import { truncateText } from "@/lib/text-utils";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import type { ClauseTypeNode, ClauseTypeTree } from "@/lib/clause-types";

interface NestedCheckboxFilterProps {
  label: string;
  data: ClauseTypeTree;
  selectedValues: string[];
  onToggle: (value: string) => void;
  labelById?: Record<string, string>;
  labelAddon?: React.ReactNode;
  className?: string;
  useModal?: boolean;
}

interface ExpandState {
  [key: string]: boolean;
}

type SearchResult = {
  key: string;
  id: string;
  path: string[];
};

type ClauseTypeValue = ClauseTypeTree[keyof ClauseTypeTree];

export function NestedCheckboxFilter({
  label,
  data,
  selectedValues,
  onToggle,
  labelById,
  labelAddon,
  className,
  useModal = false,
}: NestedCheckboxFilterProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandState, setExpandState] = useState<ExpandState>({});
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [highlightedSearchIndex, setHighlightedSearchIndex] = useState(-1);
  const labelId = useId();
  const listId = useId();
  const searchInputRef = useRef<HTMLInputElement>(null);

  const isClauseTypeNode = (
    value: ClauseTypeValue,
  ): value is ClauseTypeNode =>
    typeof value === "object" &&
    value !== null &&
    Object.prototype.hasOwnProperty.call(value, "id");

  const getChildren = (value: ClauseTypeValue): ClauseTypeTree | null => {
    if (typeof value === "string") return null;
    if (isClauseTypeNode(value)) {
      return value.children ?? null;
    }
    return value;
  };

  const getNodeId = (value: ClauseTypeValue): string | null => {
    if (typeof value === "string") return value;
    if (isClauseTypeNode(value)) return value.id;
    return null;
  };

  const isLeafNode = (value: ClauseTypeValue): boolean => {
    if (typeof value === "string") return true;
    const children = getChildren(value);
    return children === null || Object.keys(children).length === 0;
  };

  const getLeafIds = (value: ClauseTypeValue): string[] => {
    if (isLeafNode(value)) {
      const nodeId = getNodeId(value);
      return nodeId ? [nodeId] : [];
    }
    const children = getChildren(value);
    if (!children) return [];
    return Object.values(children).flatMap(getLeafIds);
  };

  const getAllSelectableIds = useMemo(() => {
    const values: string[] = [];
    const traverse = (obj: ClauseTypeTree) => {
      for (const value of Object.values(obj)) {
        values.push(...getLeafIds(value));
      }
    };
    traverse(data);
    return values;
  }, [data]);

  const totalSelected = selectedValues.length;
  const totalOptions = getAllSelectableIds.length;

  const selectedLabel = useMemo(() => {
    if (totalSelected === 0) return `All ${label}s`;
    if (totalSelected === 1) {
      const selected = selectedValues[0];
      const displayLabel = labelById?.[selected] ?? selected;
      const { truncated } = truncateText(displayLabel);
      return truncated;
    }
    if (totalSelected === totalOptions) return `All ${label}s`;
    return `${totalSelected} selected`;
  }, [label, labelById, selectedValues, totalSelected, totalOptions]);

  const searchLeafValues = useMemo(
    () => (obj: ClauseTypeTree, query: string, path: string[] = []) => {
      const results: SearchResult[] = [];
      for (const [key, value] of Object.entries(obj)) {
        if (isLeafNode(value)) {
          if (key.toLowerCase().includes(query.toLowerCase())) {
            const id = getNodeId(value);
            if (id) {
              results.push({ key, id, path: [...path] });
            }
          }
        } else {
          const children = getChildren(value);
          if (children) {
            results.push(
              ...searchLeafValues(children, query, [...path, key]),
            );
          }
        }
      }
      return results;
    },
    [],
  );

  useEffect(() => {
    if (!searchTerm.trim()) {
      setSearchResults([]);
      setHighlightedSearchIndex(-1);
      return;
    }
    const results = searchLeafValues(data, searchTerm);
    setSearchResults(results);
    setHighlightedSearchIndex(results.length > 0 ? 0 : -1);
  }, [data, searchLeafValues, searchTerm]);

  const handleSearchKeyDown = (e: React.KeyboardEvent) => {
    if (searchResults.length === 0) return;
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setHighlightedSearchIndex((prev) =>
          prev < searchResults.length - 1 ? prev + 1 : 0,
        );
        break;
      case "ArrowUp":
        e.preventDefault();
        setHighlightedSearchIndex((prev) =>
          prev > 0 ? prev - 1 : searchResults.length - 1,
        );
        break;
      case "Enter":
        e.preventDefault();
        if (highlightedSearchIndex >= 0) {
          const selectedResult = searchResults[highlightedSearchIndex];
          onToggle(selectedResult.id);
        }
        break;
      case "Escape":
        setSearchTerm("");
        setHighlightedSearchIndex(-1);
        break;
    }
  };

  const getDescendantIdsUnderPath = (
    obj: ClauseTypeTree,
    path: string[],
  ): string[] => {
    let current: ClauseTypeTree | null = obj;
    for (const pathPart of path) {
      if (!current) {
        return [];
      }
      const nextValue = current[pathPart];
      if (!nextValue) {
        return [];
      }
      const nextChildren = getChildren(nextValue);
      if (!nextChildren) {
        return [];
      }
      current = nextChildren;
    }
    const values: string[] = [];
    const collect = (node: ClauseTypeTree) => {
      for (const value of Object.values(node)) {
        values.push(...getLeafIds(value));
      }
    };
    if (current) {
      collect(current);
    }
    return values;
  };

  const selectedLeafIds = useMemo(
    () => new Set(selectedValues),
    [selectedValues],
  );

  const handleCategoryToggle = (path: string[]) => {
    let current: ClauseTypeTree | null = data;
    let currentValue: ClauseTypeValue | null = null;
    for (const pathPart of path) {
      if (!current) return;
      currentValue = current[pathPart];
      if (!currentValue) return;
      current = getChildren(currentValue);
    }
    if (!currentValue) return;
    const leafIds = getLeafIds(currentValue);
    if (leafIds.length === 0) return;
    const allSelected = leafIds.every((id) => selectedLeafIds.has(id));
    if (allSelected) {
      leafIds.forEach((id) => {
        if (selectedLeafIds.has(id)) {
          onToggle(id);
        }
      });
      return;
    }
    leafIds.forEach((id) => {
      if (!selectedLeafIds.has(id)) {
        onToggle(id);
      }
    });
  };

  const toggleExpand = (key: string) => {
    setExpandState((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const renderNestedItems = (
    obj: ClauseTypeTree,
    level: number = 0,
    path: string[] = [],
  ): JSX.Element[] => {
    const sortedKeys = Object.keys(obj).sort();
    return sortedKeys.map((key) => {
      const value = obj[key];
      const currentPath = [...path, key];
      const indentClass = level === 0 ? "" : level === 1 ? "ml-4" : "ml-8";
      const expandKey = currentPath.join(".");

      if (isLeafNode(value)) {
        const leafId = getNodeId(value);
        if (!leafId) {
          return null;
        }
        return (
          <label
            key={key}
            className={cn(
              "flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent/60 focus-within:bg-accent/60",
              indentClass,
            )}
          >
            <input
              type="checkbox"
              checked={selectedLeafIds.has(leafId)}
              onChange={() => onToggle(leafId)}
              className="h-4 w-4 rounded border-input text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-popover"
            />
            <span className="text-foreground">{key}</span>
          </label>
        );
      }

      const isExpanded = expandState[expandKey];
      const descendantIds = getDescendantIdsUnderPath(data, currentPath);
      const allSelected =
        descendantIds.length > 0 &&
        descendantIds.every((id) => selectedLeafIds.has(id));
      const someSelected = descendantIds.some((id) =>
        selectedLeafIds.has(id),
      );
      const children = getChildren(value);

      return (
        <div key={key} className={indentClass}>
          <div className="flex items-center gap-1 rounded-md px-2 py-1.5 text-sm hover:bg-accent/50">
            <button
              type="button"
              onClick={() => toggleExpand(expandKey)}
              className="inline-flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-popover"
              aria-label={isExpanded ? `Collapse ${key}` : `Expand ${key}`}
            >
              {isExpanded ? (
                <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
              )}
            </button>

            <label className="flex items-center gap-2 cursor-pointer flex-1">
              <input
                type="checkbox"
                checked={allSelected}
                ref={(input) => {
                  if (input) {
                    input.indeterminate = someSelected && !allSelected;
                  }
                }}
                onChange={() => handleCategoryToggle(currentPath)}
                className="h-4 w-4 rounded border-input text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-popover"
              />
              <span className="text-foreground font-medium">{key}</span>
            </label>
          </div>

          {isExpanded && children && (
            <div className="ml-2">
              {renderNestedItems(children, level + 1, currentPath)}
            </div>
          )}
        </div>
      );
    });
  };

  const searchResultsList = searchTerm.trim() ? (
    <div className="rounded-md border border-border/60 bg-popover">
      <Command shouldFilter={false}>
        <CommandList id={listId}>
          {searchResults.length === 0 ? (
            <CommandEmpty>
              No {label.toLowerCase()}s found matching "{searchTerm}"
            </CommandEmpty>
          ) : (
            <CommandGroup heading="Search results">
              {searchResults.map((result, index) => (
                <CommandItem
                  key={`${result.path.join(".")}.${result.key}`}
                  value={`${result.key} ${result.path.join(" ")}`}
                  onSelect={() => onToggle(result.id)}
                  className={cn(
                    "min-w-0",
                    index === highlightedSearchIndex && "bg-accent",
                  )}
                >
                  {selectedValues.includes(result.id) ? (
                    <Check className="mr-2 h-4 w-4 text-primary" aria-hidden="true" />
                  ) : (
                    <span className="mr-2 h-4 w-4" aria-hidden="true" />
                  )}
                  <div className="flex min-w-0 flex-col">
                    <span className="break-words">{result.key}</span>
                    {result.path.length > 0 && (
                      <span className="break-words text-xs text-muted-foreground">
                        {result.path.join(" 92 ")}
                      </span>
                    )}
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
          )}
        </CommandList>
      </Command>
    </div>
  ) : null;

  const content = (
    <div className="flex flex-col gap-4">
      <div className="relative">
        <div className="absolute inset-y-0 left-0 flex items-center pl-3 text-muted-foreground">
          <Search className="h-4 w-4" aria-hidden="true" />
        </div>
        <input
          ref={searchInputRef}
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onKeyDown={handleSearchKeyDown}
          aria-label={`Search ${label}`}
          placeholder="Search clause types..."
          className="block w-full rounded-md border border-input bg-background py-2 pl-10 pr-3 text-base leading-5 placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent sm:text-sm"
        />
      </div>

      {searchResultsList}

      <div className="rounded-lg border border-border/60 bg-background p-3">
        <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-3">
          Browse categories
        </div>
        <div className="max-h-[50vh] overflow-y-auto pr-1">
          {renderNestedItems(data)}
        </div>
      </div>

      {selectedValues.length > 0 && (
        <div className="rounded-lg border border-border/60 bg-muted/30 p-3">
          <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Selected ({selectedValues.length})
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {selectedValues.map((value) => (
              <Button
                key={`chip-${value}`}
                variant="outline"
                size="sm"
                onClick={() => onToggle(value)}
                className="h-7"
              >
                {labelById?.[value] ?? value}
              </Button>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex items-center justify-between gap-2">
        <span
          id={labelId}
          className="text-xs font-normal text-muted-foreground tracking-[0.15px]"
        >
          {label}
        </span>
        {labelAddon ? <div className="flex items-center">{labelAddon}</div> : null}
      </div>

      {useModal ? (
        <Dialog open={isExpanded} onOpenChange={setIsExpanded}>
          <DialogTrigger asChild>
            <button
              type="button"
              role="combobox"
              aria-expanded={isExpanded}
              aria-controls={listId}
              aria-labelledby={labelId}
              aria-haspopup="listbox"
              className={cn(
                "flex h-10 w-full items-center justify-between gap-3 rounded-md border border-input bg-background px-3 py-2 text-left text-sm text-foreground transition-colors",
                "hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              )}
            >
              <span className="truncate">{selectedLabel}</span>
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  isExpanded && "rotate-180",
                )}
                aria-hidden="true"
              />
            </button>
          </DialogTrigger>
          <DialogContent className="max-h-[90dvh] w-[calc(100vw-2rem)] max-w-[calc(100vw-2rem)] gap-4 overflow-hidden sm:w-full sm:max-w-3xl">
            <DialogHeader>
              <DialogTitle>Select {label}s</DialogTitle>
              <DialogDescription>
                Browse the tree or search for specific clause types.
              </DialogDescription>
            </DialogHeader>
            <div className="max-h-[65dvh] overflow-y-auto pr-1 pb-1 pt-1 pl-1">
              {content}
            </div>
            <div className="flex justify-end">
              <Button type="button" onClick={() => setIsExpanded(false)}>
                Done
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      ) : (
        <Popover open={isExpanded} onOpenChange={setIsExpanded}>
          <PopoverTrigger asChild>
            <button
              type="button"
              role="combobox"
              aria-expanded={isExpanded}
              aria-controls={listId}
              aria-labelledby={labelId}
              aria-haspopup="listbox"
              className={cn(
                "flex h-10 w-full items-center justify-between gap-3 rounded-md border border-input bg-background px-3 py-2 text-left text-sm text-foreground transition-colors",
                "hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              )}
            >
              <span className="truncate">{selectedLabel}</span>
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  isExpanded && "rotate-180",
                )}
                aria-hidden="true"
              />
            </button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-[420px] p-4">
            {content}
          </PopoverContent>
        </Popover>
      )}
    </div>
  );
}
