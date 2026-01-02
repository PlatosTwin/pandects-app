import { useId, useMemo, useState } from "react";
import { ChevronDown, Check } from "lucide-react";
import { truncateText } from "@/lib/text-utils";
import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";

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
  const labelId = useId();
  const listId = useId();
  const [open, setOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  const selectedOptions = useMemo(
    () => options.filter((option) => selectedValues.includes(option)),
    [options, selectedValues],
  );
  const unselectedOptions = useMemo(
    () => options.filter((option) => !selectedValues.includes(option)),
    [options, selectedValues],
  );

  const selectedLabel = useMemo(() => {
    if (selectedValues.length === 0) return `All ${label}s`;
    if (selectedValues.length === 1) {
      const { truncated } = truncateText(selectedValues[0]);
      return truncated;
    }
    return `${selectedValues.length} selected`;
  }, [label, selectedValues]);

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <label
        id={labelId}
        className="text-xs font-normal text-muted-foreground tracking-[0.15px]"
      >
        {label}
      </label>

      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            role="combobox"
            aria-expanded={open}
            aria-controls={listId}
            aria-labelledby={labelId}
            aria-haspopup="listbox"
            disabled={disabled}
            className={cn(
              "flex h-10 w-full items-center justify-between gap-3 rounded-md border border-input bg-background px-3 py-2 text-left text-sm text-foreground transition-colors",
              "hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-background",
            )}
          >
            <span className="truncate">{selectedLabel}</span>
            <ChevronDown
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform",
                open && "rotate-180",
              )}
              aria-hidden="true"
            />
          </button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-[--radix-popover-trigger-width] p-0">
          <Command shouldFilter={!hideSearch}>
            {!hideSearch && (
              <CommandInput
                placeholder={`Search ${label.toLowerCase()}s...`}
                value={searchTerm}
                onValueChange={setSearchTerm}
              />
            )}
            <CommandList id={listId}>
              <CommandEmpty>
                {searchTerm.trim()
                  ? `No ${label.toLowerCase()}s found matching "${searchTerm}"`
                  : "No options available"}
              </CommandEmpty>
              {selectedOptions.length > 0 && (
                <CommandGroup heading={`Selected (${selectedOptions.length})`}>
                  {selectedOptions.map((option) => (
                    <CommandItem
                      key={`selected-${option}`}
                      value={option}
                      onSelect={() => onToggle(option)}
                    >
                      <Check className="mr-2 h-4 w-4 text-primary" aria-hidden="true" />
                      <span className="flex-1">{option}</span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}
              <CommandGroup heading="Available">
                {unselectedOptions.map((option) => (
                  <CommandItem
                    key={`unselected-${option}`}
                    value={option}
                    onSelect={() => onToggle(option)}
                  >
                    <span className="mr-2 h-4 w-4" aria-hidden="true" />
                    <span className="flex-1">{option}</span>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
