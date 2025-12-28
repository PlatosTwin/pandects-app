import { useState, useRef, useEffect, useId } from "react";
import { DROPDOWN_ANIMATION_DELAY } from "@/lib/constants";
import { truncateText } from "@/lib/text-utils";
import { ChevronDown, ChevronUp, ChevronRight, X, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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
  const [searchResults, setSearchResults] = useState<
    Array<{ key: string; path: string[] }>
  >([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [highlightedSearchIndex, setHighlightedSearchIndex] = useState(-1);
  const dropdownId = useId();
  const modalId = useId();
  const modalTitleId = useId();
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const lastFocusedElementRef = useRef<HTMLElement | null>(null);

  // Search through leaf values (sub-sub-categories)
  const searchLeafValues = (
    obj: NestedCategory,
    query: string,
    path: string[] = [],
  ): Array<{ key: string; path: string[] }> => {
    const results: Array<{ key: string; path: string[] }> = [];

    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === "string") {
        // This is a leaf node - check if it matches the search query
        if (key.toLowerCase().includes(query.toLowerCase())) {
          results.push({ key, path: [...path] });
        }
      } else {
        // Recurse into nested object
        results.push(...searchLeafValues(value, query, [...path, key]));
      }
    }

    return results;
  };

  // Handle keyboard navigation for search results
  const handleSearchKeyDown = (e: React.KeyboardEvent) => {
    if (!showSearchResults || searchResults.length === 0) return;

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
        if (
          highlightedSearchIndex >= 0 &&
          highlightedSearchIndex < searchResults.length
        ) {
          const selectedResult = searchResults[highlightedSearchIndex];
          handleSearchResultSelect(selectedResult);
        }
        break;
      case "Escape":
        e.preventDefault();
        setSearchTerm("");
        setShowSearchResults(false);
        setHighlightedSearchIndex(-1);
        break;
    }
  };

  // Handle search input changes
  useEffect(() => {
    if (searchTerm.trim()) {
      const results = searchLeafValues(data, searchTerm);
      setSearchResults(results);
      setShowSearchResults(true);
      setHighlightedSearchIndex(results.length > 0 ? 0 : -1);
    } else {
      setSearchResults([]);
      setShowSearchResults(false);
      setHighlightedSearchIndex(-1);
    }
  }, [searchTerm, data]);

  // Handle search result selection
  const handleSearchResultSelect = (result: {
    key: string;
    path: string[];
  }) => {
    // Select the checkbox
    onToggle(result.key);

    // Expand the path to make the item visible
    const expandKeys: string[] = [];
    for (let i = 0; i < result.path.length; i++) {
      const expandKey = result.path.slice(0, i + 1).join(".");
      expandKeys.push(expandKey);
    }

    setExpandState((prev) => {
      const newState = { ...prev };
      expandKeys.forEach((key) => {
        newState[key] = true;
      });
      return newState;
    });

    // Clear search and hide results
    setSearchTerm("");
    setShowSearchResults(false);
  };

  // Initialize all categories as expanded when using modal mode
  useEffect(() => {
    if (useModal && isExpanded) {
      const initializeExpandState = (
        obj: NestedCategory,
        path: string[] = [],
      ): ExpandState => {
        const state: ExpandState = {};
        for (const [key, value] of Object.entries(obj)) {
          if (typeof value === "object") {
            const expandKey = [...path, key].join(".");
            state[expandKey] = true;
            Object.assign(
              state,
              initializeExpandState(value as NestedCategory, [...path, key]),
            );
          }
        }
        return state;
      };
      setExpandState(initializeExpandState(data));
    }
  }, [useModal, isExpanded, data]);

  // Close dropdown when clicking outside (only for dropdown mode, not modal mode)
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsExpanded(false);
      }
    }

    if (isExpanded && !useModal) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isExpanded, useModal]);

  // Reset search when modal is closed and auto-focus when opened
  useEffect(() => {
    if (!isExpanded) {
      setSearchTerm("");
      setShowSearchResults(false);
      setHighlightedSearchIndex(-1);
    } else if (useModal) {
      lastFocusedElementRef.current =
        document.activeElement as HTMLElement | null;
      // Auto-focus modal container first, then search input
      setTimeout(() => {
        if (modalRef.current) {
          modalRef.current.focus();
        }
        const focusTarget =
          searchInputRef.current ?? closeButtonRef.current ?? modalRef.current;
        focusTarget?.focus();
      }, DROPDOWN_ANIMATION_DELAY);
    }
  }, [isExpanded, useModal]);

  useEffect(() => {
    if (!isExpanded || !useModal) return;
    return () => {
      lastFocusedElementRef.current?.focus?.();
      lastFocusedElementRef.current = null;
    };
  }, [isExpanded, useModal]);

  // Get all leaf values (final clause types) from nested structure
  const getAllLeafValues = (
    obj: NestedCategory,
    path: string[] = [],
  ): string[] => {
    const values: string[] = [];

    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === "string") {
        // This is a leaf node - use the key as the final clause type
        values.push(key);
      } else {
        // Recurse into nested object
        values.push(...getAllLeafValues(value, [...path, key]));
      }
    }

    return values;
  };

  // Get all leaf values under a specific category path
  const getLeafValuesUnderPath = (
    obj: NestedCategory,
    targetPath: string[],
  ): string[] => {
    let current = obj;

    // Navigate to the target path
    for (const pathPart of targetPath) {
      if (typeof current[pathPart] === "object") {
        current = current[pathPart] as NestedCategory;
      } else {
        return [];
      }
    }

    return getAllLeafValues(current);
  };

  // Check if all children under a path are selected
  const areAllChildrenSelected = (path: string[]): boolean => {
    const childValues = getLeafValuesUnderPath(data, path);
    return (
      childValues.length > 0 &&
      childValues.every((value) => selectedValues.includes(value))
    );
  };

  // Check if some (but not all) children under a path are selected
  const areSomeChildrenSelected = (path: string[]): boolean => {
    const childValues = getLeafValuesUnderPath(data, path);
    return (
      childValues.some((value) => selectedValues.includes(value)) &&
      !areAllChildrenSelected(path)
    );
  };

  // Handle category selection (select/deselect all children)
  const handleCategoryToggle = (path: string[]) => {
    const childValues = getLeafValuesUnderPath(data, path);
    const allSelected = areAllChildrenSelected(path);

    if (allSelected) {
      // Deselect all children
      childValues.forEach((value) => {
        if (selectedValues.includes(value)) {
          onToggle(value);
        }
      });
    } else {
      // Select all children
      childValues.forEach((value) => {
        if (!selectedValues.includes(value)) {
          onToggle(value);
        }
      });
    }
  };

  // Toggle expand state for a category
  const toggleExpand = (key: string) => {
    setExpandState((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  // Render nested structure recursively
  const renderNestedItems = (
    obj: NestedCategory,
    level: number = 0,
    path: string[] = [],
  ): JSX.Element[] => {
    // Sort keys alphabetically
    const sortedKeys = Object.keys(obj).sort();

    return sortedKeys.map((key) => {
      const value = obj[key];
      const currentPath = [...path, key];
      const indentClass = level === 0 ? "" : level === 1 ? "ml-4" : "ml-8";
      const expandKey = currentPath.join(".");

      if (typeof value === "string") {
        // Leaf node - render checkbox
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
      } else {
        // Category node - render expandable group
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
                {renderNestedItems(
                  value as NestedCategory,
                  level + 1,
                  currentPath,
                )}
              </div>
            )}
          </div>
        );
      }
    });
  };

  const totalSelected = selectedValues.length;
  const totalOptions = getAllLeafValues(data).length;

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <label className="text-xs font-normal text-muted-foreground tracking-[0.15px]">
        {label}
      </label>

      <div ref={dropdownRef} className="relative">
        {/* Header showing selected count */}
        <TooltipProvider>
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            aria-expanded={isExpanded}
            aria-controls={useModal ? modalId : dropdownId}
            aria-haspopup={useModal ? "dialog" : undefined}
            className={cn(
              "flex h-10 w-full items-center justify-between gap-3 rounded-md border border-input bg-background px-3 py-2 text-left text-sm text-foreground transition-colors",
              "hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
            )}
          >
            {totalSelected === 0 ? (
              <span>{`All ${label}s`}</span>
            ) : totalSelected === 1 ? (
              (() => {
                const { truncated, needsTooltip } = truncateText(
                  selectedValues[0],
                );
                return needsTooltip ? (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span
                        tabIndex={0}
                        className="truncate focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                      >
                        {truncated}
                      </span>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">{selectedValues[0]}</p>
                    </TooltipContent>
                  </Tooltip>
                ) : (
                  <span>{truncated}</span>
                );
              })()
            ) : totalSelected === totalOptions ? (
              <span>{`All ${label}s`}</span>
            ) : (
              <span>{`${totalSelected} selected`}</span>
            )}
            {isExpanded ? (
              <ChevronUp
                className="w-4 h-4 text-muted-foreground flex-shrink-0"
                aria-hidden="true"
              />
            ) : (
              <ChevronDown
                className="w-4 h-4 text-muted-foreground flex-shrink-0"
                aria-hidden="true"
              />
            )}
          </button>
        </TooltipProvider>

        {/* Expanded nested checkbox list or Modal */}
        {isExpanded && !useModal && (
          <div
            id={dropdownId}
            className="absolute top-full left-0 right-0 z-10 mt-1 max-h-80 overflow-y-auto rounded-md border border-border bg-popover text-popover-foreground shadow-md"
          >
            <div className="p-2">{renderNestedItems(data)}</div>
          </div>
        )}
      </div>

      {/* Modal for clause types */}
      {isExpanded && useModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setIsExpanded(false)}
          />

          {/* Modal */}
          <div
            ref={modalRef}
            id={modalId}
            role="dialog"
            aria-modal="true"
            aria-labelledby={modalTitleId}
            className="relative mx-4 flex max-h-[85vh] w-full max-w-2xl flex-col rounded-lg border border-border bg-background text-foreground shadow-xl"
            onKeyDown={(e) => {
              if (e.key === "Tab") {
                const container = modalRef.current;
                if (!container) return;
                const focusable = Array.from(
                  container.querySelectorAll<HTMLElement>(
                    'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])',
                  ),
                ).filter((el) => el.getClientRects().length > 0);
                if (focusable.length === 0) return;
                const first = focusable[0];
                const last = focusable[focusable.length - 1];
                const active = document.activeElement as HTMLElement | null;
                if (e.shiftKey && active === first) {
                  e.preventDefault();
                  last.focus();
                  return;
                }
                if (!e.shiftKey && active === last) {
                  e.preventDefault();
                  first.focus();
                  return;
                }
              }
              // Handle Enter and Escape keys to close modal
              if (e.key === "Enter" || e.key === "Escape") {
                const target = e.target as HTMLElement;
                // Close modal if Enter/Escape pressed and:
                // - Search is empty, OR
                // - Not pressing on specific interactive elements (buttons, checkboxes)
                if (
                  !searchTerm.trim() ||
                  e.key === "Escape" ||
                  (target.tagName !== "BUTTON" && target.tagName !== "INPUT")
                ) {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsExpanded(false);
                }
              }
            }}
            tabIndex={-1}
          >
            {/* Header */}
            <div className="flex-shrink-0 border-b border-border p-6">
              <div className="flex items-center justify-between mb-4">
                <h3
                  id={modalTitleId}
                  className="text-lg font-medium text-foreground"
                >
                  Select {label}s
                </h3>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsExpanded(false)}
                  className="h-8 w-8"
                  ref={closeButtonRef}
                >
                  <X className="w-5 h-5" aria-hidden="true" />
                  <span className="sr-only">Close</span>
                </Button>
              </div>

              {/* Search Input */}
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                </div>
                <input
                  ref={searchInputRef}
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  onKeyDown={(e) => {
                    // Handle Enter key to close modal when search is empty
                    if (e.key === "Enter" && !searchTerm.trim()) {
                      e.preventDefault();
                      e.stopPropagation();
                      setIsExpanded(false);
                      return;
                    }
                    // Handle other keys with existing function
                    handleSearchKeyDown(e);
                  }}
                  aria-label={`Search ${label}`}
                  placeholder="Search clause types..."
                  className="block w-full rounded-md border border-input bg-background py-2 pl-10 pr-3 text-sm leading-5 placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
                />

                {/* Search Results Dropdown */}
                {showSearchResults && searchResults.length > 0 && (
                  <div className="absolute top-full left-0 right-0 z-20 mt-1 max-h-60 overflow-y-auto rounded-md border border-border bg-popover text-popover-foreground shadow-md">
                    {searchResults.map((result, index) => (
                      <button
                        type="button"
                        key={`${result.path.join(".")}.${result.key}`}
                        onClick={() => handleSearchResultSelect(result)}
                        className={cn(
                          "w-full border-b border-border px-3 py-2 text-left text-sm transition-colors last:border-b-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-popover",
                          highlightedSearchIndex === index
                            ? "bg-accent"
                            : "hover:bg-accent/50 focus-visible:bg-accent/50",
                        )}
                      >
                        <div className="text-foreground font-medium">
                          {result.key}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {result.path.join(" → ")}
                        </div>
                      </button>
                    ))}
                  </div>
                )}

                {showSearchResults &&
                  searchResults.length === 0 &&
                  searchTerm.trim() && (
                    <div className="absolute top-full left-0 right-0 z-20 mt-1 rounded-md border border-border bg-popover px-3 py-2 text-popover-foreground shadow-md">
                      <div className="text-sm text-muted-foreground">
                        No clause types found matching "{searchTerm}"
                      </div>
                    </div>
                  )}
              </div>
            </div>

            {/* Content */}
            <div className="overflow-y-auto flex-1 flex flex-col">
              {/* Selected Items (Sticky at top) */}
              {selectedValues.length > 0 && (
                <div className="flex-shrink-0 border-b border-border bg-muted/30">
                  <div className="p-4 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                    Selected ({selectedValues.length})
                  </div>
                  <div className="px-4 pb-4">
                    <div className="grid gap-2">
                      {selectedValues.map((value) => (
                        <div
                          key={`selected-${value}`}
                          className="flex items-center justify-between rounded-md border border-border bg-background p-3"
                        >
                          <span className="text-sm text-foreground font-medium">
                            {value}
                          </span>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            onClick={() => onToggle(value)}
                            className="h-8 w-8 text-muted-foreground hover:text-destructive"
                            title="Remove filter"
                          >
                            <X className="w-4 h-4" aria-hidden="true" />
                            <span className="sr-only">Remove</span>
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* All Clause Types */}
              <div className="p-6 flex-1">
                <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">
                  All Clause Types
                </div>
                {renderNestedItems(data)}
              </div>
            </div>

            {/* Footer with selection summary */}
            <div className="flex flex-shrink-0 items-center justify-between border-t border-border p-6">
              <span className="text-sm text-muted-foreground">
                {totalSelected} of {totalOptions} selected
              </span>
              <Button type="button" onClick={() => setIsExpanded(false)}>
                Done
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
