import { useState, useRef, useEffect } from "react";
import { ChevronDown, ChevronUp, ChevronRight, X } from "lucide-react";
import { cn } from "@/lib/utils";

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
  const dropdownRef = useRef<HTMLDivElement>(null);

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
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full text-left text-base font-normal text-material-text-primary bg-transparent border-none border-b border-[rgba(0,0,0,0.42)] py-2 focus:outline-none focus:border-material-blue flex items-center justify-between"
        >
          <span>
            {totalSelected === 0
              ? `All ${label}s`
              : totalSelected === 1
                ? selectedValues[0]
                : totalSelected === totalOptions
                  ? `All ${label}s`
                  : `${totalSelected} selected`}
          </span>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-material-text-secondary" />
          ) : (
            <ChevronDown className="w-4 h-4 text-material-text-secondary" />
          )}
        </button>

        {/* Bottom border line */}
        <div className="absolute bottom-0 left-0 right-0 h-px bg-[rgba(0,0,0,0.42)]" />

        {/* Expanded nested checkbox list */}
        {isExpanded && (
          <div className="absolute top-full left-0 right-0 bg-white border border-gray-200 rounded-md shadow-lg z-10 max-h-80 overflow-y-auto">
            <div className="p-2">{renderNestedItems(data)}</div>
          </div>
        )}
      </div>
    </div>
  );
}
