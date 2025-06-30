import { useState, useRef, useEffect } from "react";
import { ChevronDown, ChevronUp, ChevronRight, X, Search } from "lucide-react";
import { cn } from "@/lib/utils";
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
  tabIndex?: number;
}

interface ExpandState {
  [key: string]: boolean;
}

// Utility function to truncate text for display
const truncateText = (text: string, maxLength: number = 40) => {
  if (text.length <= maxLength) {
    return { truncated: text, needsTooltip: false };
  }
  return {
    truncated: text.substring(0, maxLength) + "...",
    needsTooltip: true,
  };
};

export function NestedCheckboxFilter({
  label,
  data,
  selectedValues,
  onToggle,
  className,
  useModal = false,
  tabIndex,
}: NestedCheckboxFilterProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandState, setExpandState] = useState<ExpandState>({});
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState<
    Array<{ key: string; path: string[] }>
  >([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [highlightedSearchIndex, setHighlightedSearchIndex] = useState(-1);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

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
        } else if (!searchTerm.trim()) {
          // Close modal when Enter is pressed with empty search
          setIsExpanded(false);
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
    } else if (useModal && searchInputRef.current) {
      // Auto-focus search input when modal opens
      setTimeout(() => {
        searchInputRef.current?.focus();
      }, 100);
    }
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
              "flex items-center gap-3 py-2 px-2 hover:bg-gray-50 cursor-pointer rounded text-sm",
              indentClass,
            )}
          >
            <input
              type="checkbox"
              checked={selectedValues.includes(key)}
              onChange={() => onToggle(key)}
              className="w-4 h-4 text-material-blue border-gray-300 rounded focus:ring-material-blue focus:ring-2"
            />
            <span className="text-material-text-primary">{key}</span>
          </label>
        );
      } else {
        // Category node - render expandable group
        const isExpanded = expandState[expandKey];
        const allSelected = areAllChildrenSelected(currentPath);
        const someSelected = areSomeChildrenSelected(currentPath);

        return (
          <div key={key} className={indentClass}>
            <div className="flex items-center gap-1 py-2 px-2 hover:bg-gray-50 rounded text-sm">
              <button
                type="button"
                onClick={() => toggleExpand(expandKey)}
                className="p-1 hover:bg-gray-200 rounded flex-shrink-0"
              >
                {isExpanded ? (
                  <ChevronDown className="w-3 h-3 text-material-text-secondary" />
                ) : (
                  <ChevronRight className="w-3 h-3 text-material-text-secondary" />
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
                  className="w-4 h-4 text-material-blue border-gray-300 rounded focus:ring-material-blue focus:ring-2"
                />
                <span className="text-material-text-primary font-medium">
                  {key}
                </span>
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
      <label className="text-xs font-normal text-material-text-secondary tracking-[0.15px]">
        {label}
      </label>

      <div ref={dropdownRef} className="relative">
        {/* Header showing selected count */}
        <TooltipProvider>
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            tabIndex={tabIndex}
            className="w-full text-left text-base font-normal text-material-text-primary bg-transparent border-none border-b border-[rgba(0,0,0,0.42)] py-2 focus:outline-none focus:border-material-blue focus:bg-blue-50 flex items-center justify-between min-h-[44px] transition-colors"
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
                      <span className="truncate">{truncated}</span>
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
              <ChevronUp className="w-4 h-4 text-material-text-secondary flex-shrink-0" />
            ) : (
              <ChevronDown className="w-4 h-4 text-material-text-secondary flex-shrink-0" />
            )}
          </button>
        </TooltipProvider>

        {/* Bottom border line */}
        <div className="absolute bottom-0 left-0 right-0 h-px bg-[rgba(0,0,0,0.42)]" />

        {/* Expanded nested checkbox list or Modal */}
        {isExpanded && !useModal && (
          <div className="absolute top-full left-0 right-0 bg-white border border-gray-200 rounded-md shadow-lg z-10 max-h-80 overflow-y-auto">
            <div className="p-2">{renderNestedItems(data)}</div>
          </div>
        )}
      </div>

      {/* Modal for clause types */}
      {isExpanded && useModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black bg-opacity-50"
            onClick={() => setIsExpanded(false)}
          />

          {/* Modal */}
          <div className="relative bg-white rounded-lg shadow-xl w-full max-w-2xl mx-4 max-h-[85vh] flex flex-col">
            {/* Header */}
            <div className="p-6 border-b border-gray-200 flex-shrink-0">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-material-text-primary">
                  Select {label}s
                </h3>
                <button
                  onClick={() => setIsExpanded(false)}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Search Input */}
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-4 w-4 text-gray-400" />
                </div>
                <input
                  ref={searchInputRef}
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  onKeyDown={handleSearchKeyDown}
                  placeholder="Search clause types..."
                  className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-material-blue focus:border-material-blue text-sm"
                />

                {/* Search Results Dropdown */}
                {showSearchResults && searchResults.length > 0 && (
                  <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-md shadow-lg z-20 max-h-60 overflow-y-auto">
                    {searchResults.map((result, index) => (
                      <button
                        key={`${result.path.join(".")}.${result.key}`}
                        onClick={() => handleSearchResultSelect(result)}
                        className={cn(
                          "w-full text-left px-3 py-2 focus:outline-none text-sm border-b border-gray-100 last:border-b-0",
                          highlightedSearchIndex === index
                            ? "bg-material-blue-light"
                            : "hover:bg-gray-50 focus:bg-gray-50",
                        )}
                      >
                        <div className="text-material-text-primary font-medium">
                          {result.key}
                        </div>
                        <div className="text-xs text-material-text-secondary mt-1">
                          {result.path.join(" → ")}
                        </div>
                      </button>
                    ))}
                  </div>
                )}

                {showSearchResults &&
                  searchResults.length === 0 &&
                  searchTerm.trim() && (
                    <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-md shadow-lg z-20 px-3 py-2">
                      <div className="text-sm text-material-text-secondary">
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
                <div className="flex-shrink-0 bg-blue-50 border-b border-gray-200">
                  <div className="p-4 text-xs font-medium text-material-text-secondary uppercase tracking-wider">
                    Selected ({selectedValues.length})
                  </div>
                  <div className="px-4 pb-4">
                    <div className="grid gap-2">
                      {selectedValues.map((value) => (
                        <div
                          key={`selected-${value}`}
                          className="flex items-center justify-between bg-white p-3 rounded border border-blue-200"
                        >
                          <span className="text-sm text-material-text-primary font-medium">
                            {value}
                          </span>
                          <button
                            type="button"
                            onClick={() => onToggle(value)}
                            className="p-1 hover:bg-red-100 rounded text-red-600 transition-colors"
                            title="Remove filter"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* All Clause Types */}
              <div className="p-6 flex-1">
                <div className="text-xs font-medium text-material-text-secondary uppercase tracking-wider mb-4">
                  All Clause Types
                </div>
                {renderNestedItems(data)}
              </div>
            </div>

            {/* Footer with selection summary */}
            <div className="flex items-center justify-between p-6 border-t border-gray-200 flex-shrink-0">
              <span className="text-sm text-material-text-secondary">
                {totalSelected} of {totalOptions} selected
              </span>
              <button
                onClick={() => setIsExpanded(false)}
                className="px-4 py-2 bg-material-blue text-white rounded hover:bg-blue-700 transition-colors"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
