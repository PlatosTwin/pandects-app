import { useState, useRef, useEffect } from "react";
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
  tabIndex?: number;
  hideSearch?: boolean;
  disabled?: boolean;
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

export function CheckboxFilter({
  label,
  options,
  selectedValues,
  onToggle,
  className,
  tabIndex,
  hideSearch = false,
  disabled = false,
}: CheckboxFilterProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [filteredOptions, setFilteredOptions] = useState<string[]>([]);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
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
          console.log("Focused component container");
        }
        if (searchInputRef.current) {
          searchInputRef.current.focus();
        }
        setHighlightedIndex(-1); // Don't auto-highlight any option
      }, 100);
    }
  }, [isExpanded]);

  // Add document-level keydown listener when dropdown is expanded
  useEffect(() => {
    if (!isExpanded) return;

    const handleDocumentKeyDown = (e: KeyboardEvent) => {
      console.log(
        "Document keydown for CheckboxFilter:",
        e.key,
        "isExpanded:",
        isExpanded,
      );

      if (e.key === "Enter" || e.key === "Escape") {
        const target = e.target as HTMLElement;
        console.log(
          "Target:",
          target.tagName,
          "searchTerm:",
          searchTerm,
          "highlighted:",
          highlightedIndex,
        );

        // Check if this event should close our dropdown
        const isInOurDropdown =
          target.closest(".absolute.top-full") === expandedDropdownRef.current;
        const isBodyTarget = target.tagName === "BODY";

        // Close dropdown if event is in our dropdown OR if target is BODY (no specific focus)
        if (isInOurDropdown || isBodyTarget) {
          console.log("Event is in our dropdown or body target");

          // Always close on Escape
          if (e.key === "Escape") {
            console.log("Closing on Escape");
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
            console.log("Closing on Enter");
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
    <div ref={componentRef} className={cn("flex flex-col gap-2", className)}>
      <label className="text-xs font-normal text-material-text-secondary tracking-[0.15px]">
        {label}
      </label>

      <div ref={dropdownRef} className="relative">
        {/* Header showing selected count or "All" */}
        <TooltipProvider>
          <button
            type="button"
            onClick={() => !disabled && setIsExpanded(!isExpanded)}
            tabIndex={disabled ? -1 : tabIndex}
            disabled={disabled}
            className={cn(
              "w-full text-left text-base font-normal bg-transparent border-none border-b py-2 flex items-center justify-between min-h-[44px] transition-colors",
              disabled
                ? "text-gray-400 border-gray-300 cursor-not-allowed"
                : "text-material-text-primary border-[rgba(0,0,0,0.42)] focus:outline-none focus:border-material-blue focus:bg-blue-50 cursor-pointer",
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
            ) : (
              <span>{`${selectedValues.length} selected`}</span>
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

        {/* Expanded dropdown with search and sticky selected items */}
        {isExpanded && !disabled && (
          <div
            ref={expandedDropdownRef}
            className="absolute top-full left-0 right-0 bg-white border border-gray-200 rounded-md shadow-lg z-10 max-h-72 flex flex-col"
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
              <div className="p-3 border-b border-gray-200">
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Search className="h-4 w-4 text-gray-400" />
                  </div>
                  <input
                    ref={searchInputRef}
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={`Search ${label.toLowerCase()}s...`}
                    className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-material-blue focus:border-material-blue text-sm"
                  />
                </div>
              </div>
            )}

            {/* Selected Items (Sticky at top) */}
            {selectedOptions.length > 0 && (
              <div className="flex-shrink-0 bg-blue-50 border-b border-gray-200">
                <div className="p-2 text-xs font-medium text-material-text-secondary uppercase tracking-wider">
                  Selected ({selectedOptions.length})
                </div>
                <div className="p-2 pt-0">
                  {selectedOptions.map((option) => (
                    <label
                      key={`selected-${option}`}
                      className="flex items-center gap-3 py-2 px-2 hover:bg-blue-100 cursor-pointer rounded text-sm"
                    >
                      <input
                        type="checkbox"
                        checked={true}
                        onChange={() => onToggle(option)}
                        className="w-4 h-4 text-material-blue border-gray-300 rounded focus:ring-material-blue focus:ring-2"
                      />
                      <span className="text-material-text-primary font-medium">
                        {option}
                      </span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          onToggle(option);
                        }}
                        className="ml-auto p-1 hover:bg-blue-200 rounded"
                        title="Remove filter"
                      >
                        <X className="w-3 h-3 text-gray-500" />
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
                      <div className="p-2 text-xs font-medium text-material-text-secondary uppercase tracking-wider">
                        Search Results
                      </div>
                      {filteredOptions.map((option, index) => (
                        <label
                          key={`filtered-${option}`}
                          className={cn(
                            "flex items-center gap-3 py-2 px-2 cursor-pointer rounded text-sm",
                            highlightedIndex === index
                              ? "bg-material-blue-light"
                              : "hover:bg-gray-50",
                          )}
                        >
                          <input
                            type="checkbox"
                            checked={selectedValues.includes(option)}
                            onChange={() => onToggle(option)}
                            className="w-4 h-4 text-material-blue border-gray-300 rounded focus:ring-material-blue focus:ring-2"
                          />
                          <span className="text-material-text-primary">
                            {option}
                          </span>
                        </label>
                      ))}
                    </>
                  ) : (
                    <div className="p-4 text-center text-sm text-material-text-secondary">
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
                            "flex items-center gap-3 py-2 px-2 cursor-pointer rounded text-sm",
                            !searchTerm.trim() && highlightedIndex === index
                              ? "bg-material-blue-light"
                              : "hover:bg-gray-50",
                          )}
                        >
                          <input
                            type="checkbox"
                            checked={false}
                            onChange={() => onToggle(option)}
                            className="w-4 h-4 text-material-blue border-gray-300 rounded focus:ring-material-blue focus:ring-2"
                          />
                          <span className="text-material-text-primary">
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
