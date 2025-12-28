import { useState, useRef, useEffect, useId } from "react";
import { DROPDOWN_ANIMATION_DELAY } from "@/lib/constants";
import { truncateText } from "@/lib/text-utils";
import { ChevronDown, ChevronUp, Search, X } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface CheckboxFilterProps {
  label: string;
  options: string[];
  selectedValues: string[];
  onToggle: (value: string) => void;
  className?: string;
  hideSearch?: boolean;
  disabled?: boolean;
}

export function CheckboxFilter({
  label,
  options,
  selectedValues,
  onToggle,
  className,
  hideSearch = false,
  disabled = false,
}: CheckboxFilterProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [filteredOptions, setFilteredOptions] = useState<string[]>([]);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const dropdownId = useId();
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const expandedDropdownRef = useRef<HTMLDivElement>(null);

  // Filter options based on search term
  useEffect(() => {
    if (searchTerm.trim()) {
      const filtered = options.filter((option) =>
        option.toLowerCase().includes(searchTerm.toLowerCase()),
      );
      setFilteredOptions(filtered);
      setHighlightedIndex(filtered.length > 0 ? 0 : -1);
    } else {
      setFilteredOptions([]);
      setHighlightedIndex(-1);
    }
  }, [searchTerm, options]);

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isExpanded) return;

    // Determine which options to navigate through
    const navigableOptions = searchTerm.trim()
      ? filteredOptions
      : unselectedOptions;

    if (navigableOptions.length === 0) return;

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setHighlightedIndex((prev) => {
          if (prev === -1) return 0; // Start from first item if nothing highlighted
          return prev < navigableOptions.length - 1 ? prev + 1 : 0;
        });
        break;
      case "ArrowUp":
        e.preventDefault();
        setHighlightedIndex((prev) => {
          if (prev === -1) return navigableOptions.length - 1; // Start from last item if nothing highlighted
          return prev > 0 ? prev - 1 : navigableOptions.length - 1;
        });
        break;
      case "Enter":
        e.preventDefault();
        if (
          highlightedIndex >= 0 &&
          highlightedIndex < navigableOptions.length
        ) {
          const selectedOption = navigableOptions[highlightedIndex];
          onToggle(selectedOption);
          // Clear search term after selection and reset highlight
          setSearchTerm("");
          setHighlightedIndex(-1);
          // Keep dropdown open for multiple selections
        } else {
          // Close dropdown when no item is highlighted
          setSearchTerm("");
          setIsExpanded(false);
        }
        break;
      case "Escape":
        e.preventDefault();
        setSearchTerm("");
        setIsExpanded(false);
        break;
    }
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsExpanded(false);
        setSearchTerm("");
      }
    }

    if (isExpanded) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isExpanded]);

  // Separate selected and unselected options for sticky behavior
  const selectedOptions = options.filter((option) =>
    selectedValues.includes(option),
  );
  const unselectedOptions = options.filter(
    (option) => !selectedValues.includes(option),
  );

  // Component container ref for focus
  const componentRef = useRef<HTMLDivElement>(null);

  // Reset search when closing
  useEffect(() => {
    if (!isExpanded) {
      setSearchTerm("");
      setHighlightedIndex(-1);
    } else {
      // Focus component container first so it can capture keydown events
      setTimeout(() => {
        if (componentRef.current) {
          componentRef.current.focus();
        }
        if (searchInputRef.current) {
          searchInputRef.current.focus();
        }
        setHighlightedIndex(-1); // Don't auto-highlight any option
      }, DROPDOWN_ANIMATION_DELAY);
    }
  }, [isExpanded]);

  // Add document-level keydown listener when dropdown is expanded
  useEffect(() => {
    if (!isExpanded) return;

    const handleDocumentKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === "Escape") {
        const target = e.target as HTMLElement;

        // Check if this event should close our dropdown
        const isInOurDropdown =
          target.closest(".absolute.top-full") === expandedDropdownRef.current;
        const isBodyTarget = target.tagName === "BODY";

        // Close dropdown if event is in our dropdown OR if target is BODY (no specific focus)
        if (isInOurDropdown || isBodyTarget) {
          // Always close on Escape
          if (e.key === "Escape") {
            e.preventDefault();
            e.stopPropagation();
            setIsExpanded(false);
            setSearchTerm("");
            return;
          }

          // Close on Enter if not actively using search input or if target is BODY
          if (
            target.tagName !== "INPUT" ||
            (!searchTerm.trim() && highlightedIndex === -1) ||
            isBodyTarget
          ) {
            e.preventDefault();
            e.stopPropagation();
            setIsExpanded(false);
            setSearchTerm("");
          }
        }
      }
    };

    document.addEventListener("keydown", handleDocumentKeyDown, true); // Use capture phase
    return () => {
      document.removeEventListener("keydown", handleDocumentKeyDown, true);
    };
  }, [isExpanded, searchTerm, highlightedIndex]);

  return (
    <div
      ref={componentRef}
      tabIndex={-1}
      className={cn("flex flex-col gap-2", className)}
    >
      <label className="text-xs font-normal text-muted-foreground tracking-[0.15px]">
        {label}
      </label>

      <div ref={dropdownRef} className="relative">
        {/* Header showing selected count or "All" */}
        <TooltipProvider>
          <button
            type="button"
            onClick={() => !disabled && setIsExpanded(!isExpanded)}
            aria-expanded={isExpanded}
            aria-controls={dropdownId}
            disabled={disabled}
            className={cn(
              "flex h-10 w-full items-center justify-between gap-3 rounded-md border border-input bg-background px-3 py-2 text-left text-sm text-foreground transition-colors",
              "hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-background",
            )}
          >
            {selectedValues.length === 0 ? (
              <span>{`All ${label}s`}</span>
            ) : selectedValues.length === 1 ? (
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
            ) : (
              <span>{`${selectedValues.length} selected`}</span>
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

        {/* Expanded dropdown with search and sticky selected items */}
        {isExpanded && !disabled && (
          <div
            ref={expandedDropdownRef}
            id={dropdownId}
            className="absolute top-full left-0 right-0 z-10 mt-1 flex max-h-72 flex-col overflow-hidden rounded-md border border-border bg-popover text-popover-foreground shadow-md"
            onKeyDown={(e) => {
              // Handle Enter and Escape keys to close dropdown
              if (e.key === "Enter" || e.key === "Escape") {
                const target = e.target as HTMLElement;
                // Always close on Escape
                if (e.key === "Escape") {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsExpanded(false);
                  setSearchTerm("");
                  return;
                }
                // Close on Enter if not in search input, or if search is empty and nothing highlighted
                if (
                  target.tagName !== "INPUT" ||
                  (!searchTerm.trim() && highlightedIndex === -1)
                ) {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsExpanded(false);
                  setSearchTerm("");
                }
              }
            }}
            tabIndex={-1}
          >
            {/* Search Input */}
            {!hideSearch && (
              <div className="border-b border-border p-2">
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Search className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                  </div>
                  <input
                    ref={searchInputRef}
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    onKeyDown={handleKeyDown}
                    aria-label={`Search ${label}`}
                    placeholder={`Search ${label.toLowerCase()}s...`}
                    className="block w-full rounded-md border border-input bg-background py-2 pl-10 pr-3 text-sm leading-5 placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
                  />
                </div>
              </div>
            )}

            {/* Selected Items (Sticky at top) */}
            {selectedOptions.length > 0 && (
              <div className="flex-shrink-0 border-b border-border bg-muted/30">
                <div className="px-3 py-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Selected ({selectedOptions.length})
                </div>
                <div className="px-2 pb-2">
                  {selectedOptions.map((option) => (
                    <label
                      key={`selected-${option}`}
                      className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent focus-within:bg-accent"
                    >
                      <input
                        type="checkbox"
                        checked={true}
                        onChange={() => onToggle(option)}
                        className="h-4 w-4 rounded border-input text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-popover"
                      />
                      <span className="text-foreground font-medium">
                        {option}
                      </span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          onToggle(option);
                        }}
                        aria-label={`Remove ${option}`}
                        className="ml-auto inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-popover"
                        title="Remove filter"
                      >
                        <X className="h-3.5 w-3.5" aria-hidden="true" />
                      </button>
                    </label>
                  ))}
                </div>
              </div>
            )}

            {/* Search Results or All Unselected Options */}
            <div className="flex-1 overflow-y-auto">
              {!hideSearch && searchTerm.trim() ? (
                // Show filtered search results
                <div className="p-2">
                  {filteredOptions.length > 0 ? (
                    <>
                      <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Search Results
                      </div>
                      {filteredOptions.map((option, index) => (
                        <label
                          key={`filtered-${option}`}
                          className={cn(
                            "flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent focus-within:bg-accent",
                            highlightedIndex === index
                              ? "bg-accent"
                              : null,
                          )}
                        >
                          <input
                            type="checkbox"
                            checked={selectedValues.includes(option)}
                            onChange={() => onToggle(option)}
                            className="h-4 w-4 rounded border-input text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-popover"
                          />
                          <span className="text-foreground">
                            {option}
                          </span>
                        </label>
                      ))}
                    </>
                  ) : (
                    <div className="p-4 text-center text-sm text-muted-foreground">
                      No {label.toLowerCase()}s found matching "{searchTerm}"
                    </div>
                  )}
                </div>
              ) : (
                // Show all unselected options when not searching
                <div className="p-2">
                  {unselectedOptions.length > 0 && (
                    <>
                      {unselectedOptions.map((option, index) => (
                        <label
                          key={`unselected-${option}`}
                          className={cn(
                            "flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent focus-within:bg-accent",
                            !searchTerm.trim() && highlightedIndex === index
                              ? "bg-accent"
                              : null,
                          )}
                        >
                          <input
                            type="checkbox"
                            checked={false}
                            onChange={() => onToggle(option)}
                            className="h-4 w-4 rounded border-input text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-popover"
                          />
                          <span className="text-foreground">
                            {option}
                          </span>
                        </label>
                      ))}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
