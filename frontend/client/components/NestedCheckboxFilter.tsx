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

interface NestedCategory {
  [key: string]: NestedCategory | string;
}

interface NestedCheckboxFilterProps {
  label: string;
  data: NestedCategory;
  selectedValues: string[];
  onToggle: (value: string) => void;
  className?: string;
  useModal?: boolean;
}

interface ExpandState {
  [key: string]: boolean;
}

type SearchResult = {
  key: string;
  path: string[];
};

export function NestedCheckboxFilter({
  label,
  data,
  selectedValues,
  onToggle,
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

  const getAllLeafValues = useMemo(() => {
    const values: string[] = [];
    const traverse = (obj: NestedCategory) => {
      for (const [key, value] of Object.entries(obj)) {
        if (typeof value === "string") {
          values.push(key);
        } else {
          traverse(value as NestedCategory);
        }
      }
    };
    traverse(data);
    return values;
  }, [data]);

  const totalSelected = selectedValues.length;
  const totalOptions = getAllLeafValues.length;

  const selectedLabel = useMemo(() => {
    if (totalSelected === 0) return `All ${label}s`;
    if (totalSelected === 1) {
      const { truncated } = truncateText(selectedValues[0]);
      return truncated;
    }
    if (totalSelected === totalOptions) return `All ${label}s`;
    return `${totalSelected} selected`;
  }, [label, selectedValues, totalSelected, totalOptions]);

  const searchLeafValues = useMemo(
    () => (obj: NestedCategory, query: string, path: string[] = []) => {
      const results: SearchResult[] = [];
      for (const [key, value] of Object.entries(obj)) {
        if (typeof value === "string") {
          if (key.toLowerCase().includes(query.toLowerCase())) {
            results.push({ key, path: [...path] });
          }
        } else {
          results.push(...searchLeafValues(value as NestedCategory, query, [...path, key]));
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
          onToggle(selectedResult.key);
        }
        break;
      case "Escape":
        setSearchTerm("");
        setHighlightedSearchIndex(-1);
        break;
    }
  };

  const getLeafValuesUnderPath = (obj: NestedCategory, path: string[]): string[] => {
    let current = obj;
    for (const pathPart of path) {
      if (typeof current[pathPart] === "object") {
        current = current[pathPart] as NestedCategory;
      } else {
        return [];
      }
    }
    const values: string[] = [];
    const collect = (node: NestedCategory) => {
      for (const [key, value] of Object.entries(node)) {
        if (typeof value === "string") values.push(key);
        else collect(value as NestedCategory);
      }
    };
    collect(current);
    return values;
  };

  const areAllChildrenSelected = (path: string[]) => {
    const values = getLeafValuesUnderPath(data, path);
    return values.length > 0 && values.every((value) => selectedValues.includes(value));
  };

  const areSomeChildrenSelected = (path: string[]) => {
    const values = getLeafValuesUnderPath(data, path);
    return values.some((value) => selectedValues.includes(value)) && !areAllChildrenSelected(path);
  };

  const handleCategoryToggle = (path: string[]) => {
    const values = getLeafValuesUnderPath(data, path);
    const allSelected = values.length > 0 && values.every((value) => selectedValues.includes(value));
    values.forEach((value) => {
      const hasValue = selectedValues.includes(value);
      if (allSelected ? hasValue : !hasValue) {
        onToggle(value);
      }
    });
  };

  const toggleExpand = (key: string) => {
    setExpandState((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const renderNestedItems = (
    obj: NestedCategory,
    level: number = 0,
    path: string[] = [],
  ): JSX.Element[] => {
    const sortedKeys = Object.keys(obj).sort();
    return sortedKeys.map((key) => {
      const value = obj[key];
      const currentPath = [...path, key];
      const indentClass = level === 0 ? "" : level === 1 ? "ml-4" : "ml-8";
      const expandKey = currentPath.join(".");

      if (typeof value === "string") {
        return (
          <label
            key={key}
            className={cn(
              "flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent focus-within:bg-accent",
              indentClass,
            )}
          >
            <input
              type="checkbox"
              checked={selectedValues.includes(key)}
              onChange={() => onToggle(key)}
              className="h-4 w-4 rounded border-input text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-popover"
            />
            <span className="text-foreground">{key}</span>
          </label>
        );
      }

      const isExpanded = expandState[expandKey];
      const allSelected = areAllChildrenSelected(currentPath);
      const someSelected = areSomeChildrenSelected(currentPath);

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
                    input.indeterminate = someSelected;
                  }
                }}
                onChange={() => handleCategoryToggle(currentPath)}
                className="h-4 w-4 rounded border-input text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-popover"
              />
              <span className="text-foreground font-medium">{key}</span>
            </label>
          </div>

          {isExpanded && (
            <div className="ml-2">
              {renderNestedItems(value as NestedCategory, level + 1, currentPath)}
            </div>
          )}
        </div>
      );
    });
  };

  const searchResultsList = searchTerm.trim() ? (
    <div className="rounded-md border border-border bg-popover">
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
                  onSelect={() => onToggle(result.key)}
                  className={cn(
                    index === highlightedSearchIndex && "bg-accent",
                  )}
                >
                  {selectedValues.includes(result.key) ? (
                    <Check className="mr-2 h-4 w-4 text-primary" aria-hidden="true" />
                  ) : (
                    <span className="mr-2 h-4 w-4" aria-hidden="true" />
                  )}
                  <div className="flex flex-col">
                    <span>{result.key}</span>
                    {result.path.length > 0 && (
                      <span className="text-xs text-muted-foreground">
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
          className="block w-full rounded-md border border-input bg-background py-2 pl-10 pr-3 text-sm leading-5 placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
        />
      </div>

      {searchResultsList}

      <div className="rounded-lg border border-border bg-background p-3">
        <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-3">
          Browse categories
        </div>
        <div className="max-h-[50vh] overflow-y-auto pr-1">
          {renderNestedItems(data)}
        </div>
      </div>

      {selectedValues.length > 0 && (
        <div className="rounded-lg border border-border bg-muted/30 p-3">
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
                {value}
              </Button>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <label
        id={labelId}
        className="text-xs font-normal text-muted-foreground tracking-[0.15px]"
      >
        {label}
      </label>

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
          <DialogContent className="max-w-3xl max-h-[90dvh] gap-4 overflow-hidden">
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
