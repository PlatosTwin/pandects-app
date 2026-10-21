import { useId, useMemo, useState, type KeyboardEvent, type ReactNode } from "react";
import { Check, ChevronDown, ChevronRight, Search, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import type { ClauseTypeNode, ClauseTypeTree } from "@/lib/clause-types";

type TreeValue = ClauseTypeTree[string];

function isNode(value: TreeValue): value is ClauseTypeNode {
  return typeof value === "object" && value !== null && Object.prototype.hasOwnProperty.call(value, "id");
}

function childrenOf(value: TreeValue): ClauseTypeTree | null {
  if (typeof value === "string") return null;
  if (isNode(value)) return value.children ?? null;
  return value;
}

function nodeIdOf(value: TreeValue): string | null {
  if (typeof value === "string") return value;
  if (isNode(value)) return value.id;
  return null;
}

function isLeaf(value: TreeValue): boolean {
  const children = childrenOf(value);
  return children === null || Object.keys(children).length === 0;
}

/** Selectable ids under a tree value (the value itself when it is a leaf). */
export function leafIdsOf(value: TreeValue): string[] {
  if (isLeaf(value)) {
    const id = nodeIdOf(value);
    return id ? [id] : [];
  }
  return Object.values(childrenOf(value) ?? {}).flatMap(leafIdsOf);
}

interface LeafMatch {
  key: string;
  id: string;
  path: string[];
}

function searchLeaves(tree: ClauseTypeTree, query: string, path: string[] = []): LeafMatch[] {
  const needle = query.toLowerCase();
  const matches: LeafMatch[] = [];
  for (const [key, value] of Object.entries(tree)) {
    if (isLeaf(value)) {
      const id = nodeIdOf(value);
      if (id && (key.toLowerCase().includes(needle) || path.join(" ").toLowerCase().includes(needle))) {
        matches.push({ key, id, path });
      }
    } else {
      const children = childrenOf(value);
      if (children) matches.push(...searchLeaves(children, query, [...path, key]));
    }
  }
  return matches;
}

export interface NestedCheckboxTreeProps {
  data: ClauseTypeTree;
  selectedValues: string[];
  onToggle: (value: string) => void;
  labelById?: Record<string, string>;
  /** Plural noun for placeholders, e.g. "clause types" or "tax clause types". */
  noun: string;
  /** Clears the whole selection. */
  onClear?: () => void;
  className?: string;
}

/**
 * Searchable taxonomy tree: a type-ahead over leaf labels, a browsable tree
 * with tri-state category checkboxes, and the current selection as chips.
 */
export function NestedCheckboxTree({
  data,
  selectedValues,
  onToggle,
  labelById,
  noun,
  onClear,
  className,
}: NestedCheckboxTreeProps) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [searchTerm, setSearchTerm] = useState("");
  const [highlighted, setHighlighted] = useState(0);
  const listId = useId();
  const selected = useMemo(() => new Set(selectedValues), [selectedValues]);
  const matches = useMemo(
    () => (searchTerm.trim() ? searchLeaves(data, searchTerm.trim()) : []),
    [data, searchTerm],
  );

  const toggleCategory = (value: TreeValue) => {
    const ids = leafIdsOf(value);
    if (ids.length === 0) return;
    const allSelected = ids.every((id) => selected.has(id));
    ids.forEach((id) => {
      if (allSelected ? selected.has(id) : !selected.has(id)) onToggle(id);
    });
  };

  const onSearchKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Escape" && searchTerm) {
      // Clear the query first; a second Escape closes the popover.
      event.stopPropagation();
      setSearchTerm("");
      return;
    }
    if (matches.length === 0) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setHighlighted((prev) => (prev + 1) % matches.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlighted((prev) => (prev - 1 + matches.length) % matches.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      const match = matches[Math.min(highlighted, matches.length - 1)];
      if (match) onToggle(match.id);
    }
  };

  const renderTree = (tree: ClauseTypeTree, level: number, path: string[]): ReactNode[] =>
    Object.keys(tree)
      .sort()
      .map((key) => {
        const value = tree[key];
        const expandKey = [...path, key].join(".");
        const indent = level === 0 ? "" : "ml-5";
        if (isLeaf(value)) {
          const id = nodeIdOf(value);
          if (!id) return null;
          return (
            <label
              key={expandKey}
              className={cn(
                "flex cursor-pointer items-start gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent/60 focus-within:bg-accent/60",
                indent,
              )}
            >
              <input
                type="checkbox"
                checked={selected.has(id)}
                onChange={() => onToggle(id)}
                className="mt-0.5 h-4 w-4 shrink-0 rounded border-input accent-primary"
              />
              <span className="text-foreground">{key}</span>
            </label>
          );
        }
        const ids = leafIdsOf(value);
        const allSelected = ids.length > 0 && ids.every((id) => selected.has(id));
        const someSelected = ids.some((id) => selected.has(id));
        const isOpen = Boolean(expanded[expandKey]);
        const children = childrenOf(value);
        return (
          <div key={expandKey} className={indent}>
            <div className="flex items-center gap-1 rounded-md px-1 py-1 text-sm hover:bg-accent/50">
              <button
                type="button"
                onClick={() => setExpanded((prev) => ({ ...prev, [expandKey]: !prev[expandKey] }))}
                className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                aria-expanded={isOpen}
                aria-label={isOpen ? `Collapse ${key}` : `Expand ${key}`}
              >
                {isOpen ? (
                  <ChevronDown className="h-4 w-4" aria-hidden="true" />
                ) : (
                  <ChevronRight className="h-4 w-4" aria-hidden="true" />
                )}
              </button>
              <label className="flex flex-1 cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={allSelected}
                  ref={(input) => {
                    if (input) input.indeterminate = someSelected && !allSelected;
                  }}
                  onChange={() => toggleCategory(value)}
                  className="h-4 w-4 shrink-0 rounded border-input accent-primary"
                />
                <span className="font-medium text-foreground">{key}</span>
              </label>
            </div>
            {isOpen && children ? <div>{renderTree(children, level + 1, [...path, key])}</div> : null}
          </div>
        );
      });

  return (
    <div className={cn("flex min-h-0 flex-col gap-3", className)}>
      <div className="relative">
        <Search
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden="true"
        />
        <input
          type="text"
          value={searchTerm}
          onChange={(event) => {
            setSearchTerm(event.target.value);
            setHighlighted(0);
          }}
          onKeyDown={onSearchKeyDown}
          aria-label={`Search ${noun}`}
          aria-controls={listId}
          placeholder={`Search ${noun}…`}
          className="block h-10 w-full rounded-md border border-input bg-background pl-9 pr-3 text-base placeholder:text-muted-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:text-sm"
        />
      </div>

      <div id={listId} className="min-h-0 flex-1 overflow-y-auto overscroll-contain pr-1">
        {searchTerm.trim() ? (
          matches.length === 0 ? (
            <p className="px-2 py-6 text-center text-sm text-muted-foreground">
              No {noun} match “{searchTerm.trim()}”.
            </p>
          ) : (
            <ul aria-label={`Matching ${noun}`}>
              {matches.map((match, index) => {
                const isSelected = selected.has(match.id);
                return (
                  <li key={`${match.path.join(".")}.${match.key}`}>
                    <button
                      type="button"
                      aria-pressed={isSelected}
                      onClick={() => onToggle(match.id)}
                      onMouseEnter={() => setHighlighted(index)}
                      className={cn(
                        "flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                        index === highlighted && "bg-accent",
                      )}
                    >
                      <Check
                        className={cn("mt-0.5 h-4 w-4 shrink-0 text-primary", !isSelected && "invisible")}
                        aria-hidden="true"
                      />
                      <span className="flex min-w-0 flex-col">
                        <span className="break-words text-foreground">{match.key}</span>
                        {match.path.length > 0 ? (
                          <span className="break-words text-xs text-muted-foreground">
                            {match.path.join(" › ")}
                          </span>
                        ) : null}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )
        ) : (
          renderTree(data, 0, [])
        )}
      </div>

      {selectedValues.length > 0 ? (
        <div className="border-t border-border pt-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-muted-foreground">
              Selected ({selectedValues.length})
            </span>
            {onClear ? (
              <button
                type="button"
                onClick={onClear}
                className="text-xs font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                Clear
              </button>
            ) : null}
          </div>
          <div className="flex max-h-24 flex-wrap gap-1.5 overflow-y-auto">
            {selectedValues.map((value) => {
              const label = labelById?.[value] ?? "Unknown clause type";
              return (
                <button
                  key={value}
                  type="button"
                  onClick={() => onToggle(value)}
                  className="inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-muted/40 px-2 py-1 text-xs text-foreground hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label={`Remove ${label}`}
                >
                  <span className="truncate">{label}</span>
                  <X className="h-3 w-3 shrink-0" aria-hidden="true" />
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export interface NestedCheckboxFilterProps extends Omit<NestedCheckboxTreeProps, "className"> {
  /** Accessible name and visible label of the trigger. */
  label: string;
  /** Text when nothing is selected, e.g. "All clause types". */
  placeholder: string;
  isLoading?: boolean;
  disabled?: boolean;
  className?: string;
  triggerClassName?: string;
}

/** Trigger button + popover around `NestedCheckboxTree`. */
export function NestedCheckboxFilter({
  label,
  placeholder,
  isLoading = false,
  disabled = false,
  className,
  triggerClassName,
  ...treeProps
}: NestedCheckboxFilterProps) {
  const [open, setOpen] = useState(false);
  const { selectedValues, labelById } = treeProps;
  const summary =
    selectedValues.length === 0
      ? placeholder
      : selectedValues.length === 1
        ? (labelById?.[selectedValues[0]] ?? "1 selected")
        : `${labelById?.[selectedValues[0]] ?? "1 type"} +${selectedValues.length - 1}`;

  return (
    <div className={className}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            disabled={disabled}
            aria-label={`${label}: ${summary}`}
            className={cn(
              "flex h-10 w-full min-w-0 items-center justify-between gap-2 rounded-md border border-input bg-background px-3 text-left text-sm text-foreground transition-colors",
              "hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50",
              selectedValues.length === 0 && "text-muted-foreground",
              triggerClassName,
            )}
          >
            <span className="truncate">{isLoading && selectedValues.length === 0 ? "Loading…" : summary}</span>
            <ChevronDown
              className={cn("h-4 w-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")}
              aria-hidden="true"
            />
          </button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          aria-label={label}
          className="flex max-h-[min(32rem,var(--radix-popover-content-available-height))] w-[min(28rem,calc(100vw-2rem))] flex-col p-3"
        >
          <NestedCheckboxTree {...treeProps} className="min-h-0 flex-1" />
        </PopoverContent>
      </Popover>
    </div>
  );
}
