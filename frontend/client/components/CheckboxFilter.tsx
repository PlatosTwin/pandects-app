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
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsExpanded(false);
      }
    }

    if (isExpanded) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isExpanded]);

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
