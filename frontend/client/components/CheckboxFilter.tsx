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

        {/* Expanded checkbox list */}
        {isExpanded && (
          <div className="absolute top-full left-0 right-0 bg-white border border-gray-200 rounded-md shadow-lg z-10 max-h-48 overflow-y-auto">
            <div className="p-2">
              {options.map((option) => (
                <label
                  key={option}
                  className="flex items-center gap-3 py-2 px-2 hover:bg-gray-50 cursor-pointer rounded text-sm"
                >
                  <input
                    type="checkbox"
                    checked={selectedValues.includes(option)}
                    onChange={() => onToggle(option)}
                    className="w-4 h-4 text-material-blue border-gray-300 rounded focus:ring-material-blue focus:ring-2"
                  />
                  <span className="text-material-text-primary">{option}</span>
                </label>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
