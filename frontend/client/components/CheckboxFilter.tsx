import { useState, useRef, useEffect } from "react";
import { ChevronDown, ChevronUp, Search, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface CheckboxFilterProps {
  label: string;
  options: string[];
  selectedValues: string[];
  onToggle: (value: string) => void;
  className?: string;
}

export function CheckboxFilter({
  label,
  options,
  selectedValues,
  onToggle,
  className,
}: CheckboxFilterProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [filteredOptions, setFilteredOptions] = useState<string[]>([]);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Filter options based on search term
  useEffect(() => {
    if (searchTerm.trim()) {
      const filtered = options.filter((option) =>
        option.toLowerCase().includes(searchTerm.toLowerCase()),
      );
      setFilteredOptions(filtered);
      setHighlightedIndex(0);
    } else {
      setFilteredOptions([]);
      setHighlightedIndex(-1);
    }
  }, [searchTerm, options]);

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isExpanded || filteredOptions.length === 0) return;

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setHighlightedIndex((prev) =>
          prev < filteredOptions.length - 1 ? prev + 1 : 0,
        );
        break;
      case "ArrowUp":
        e.preventDefault();
        setHighlightedIndex((prev) =>
          prev > 0 ? prev - 1 : filteredOptions.length - 1,
        );
        break;
      case "Enter":
        e.preventDefault();
        if (
          highlightedIndex >= 0 &&
          highlightedIndex < filteredOptions.length
        ) {
          const selectedOption = filteredOptions[highlightedIndex];
          onToggle(selectedOption);
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

  // Reset search when closing
  useEffect(() => {
    if (!isExpanded) {
      setSearchTerm("");
      setHighlightedIndex(-1);
    } else if (searchInputRef.current) {
      // Focus search input when opening
      setTimeout(() => searchInputRef.current?.focus(), 100);
    }
  }, [isExpanded]);

  // Separate selected and unselected options for sticky behavior
  const selectedOptions = options.filter((option) =>
    selectedValues.includes(option),
  );
  const unselectedOptions = options.filter(
    (option) => !selectedValues.includes(option),
  );

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <label className="text-xs font-normal text-material-text-secondary tracking-[0.15px]">
        {label}
      </label>

      <div ref={dropdownRef} className="relative">
        {/* Header showing selected count or "All" */}
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full text-left text-base font-normal text-material-text-primary bg-transparent border-none border-b border-[rgba(0,0,0,0.42)] py-2 focus:outline-none focus:border-material-blue flex items-center justify-between"
        >
          <span>
            {selectedValues.length === 0
              ? `All ${label}s`
              : selectedValues.length === 1
                ? selectedValues[0]
                : `${selectedValues.length} selected`}
          </span>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-material-text-secondary" />
          ) : (
            <ChevronDown className="w-4 h-4 text-material-text-secondary" />
          )}
        </button>

        {/* Bottom border line */}
        <div className="absolute bottom-0 left-0 right-0 h-px bg-[rgba(0,0,0,0.42)]" />

        {/* Expanded dropdown with search and sticky selected items */}
        {isExpanded && (
          <div className="absolute top-full left-0 right-0 bg-white border border-gray-200 rounded-md shadow-lg z-10 max-h-72 flex flex-col">
            {/* Search Input */}
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
              {searchTerm.trim() ? (
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
                      <div className="p-2 text-xs font-medium text-material-text-secondary uppercase tracking-wider">
                        Available
                      </div>
                      {unselectedOptions.map((option) => (
                        <label
                          key={`unselected-${option}`}
                          className="flex items-center gap-3 py-2 px-2 hover:bg-gray-50 cursor-pointer rounded text-sm"
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
