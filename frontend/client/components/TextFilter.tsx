import { useId, useState } from "react";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";

interface TextFilterProps {
  label: string;
  value: string;
  /** Called with the trimmed value on Enter or blur (not on every keystroke). */
  onCommit: (value: string) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

/**
 * Free-text filter (agreement / section UUID). Edits stay local until Enter
 * or blur, so a half-typed UUID never runs a search. Remount with a `key` to
 * pick up an outside change to `value`.
 */
export function TextFilter({
  label,
  value,
  onCommit,
  placeholder,
  className,
  disabled = false,
}: TextFilterProps) {
  const inputId = useId();
  const [draft, setDraft] = useState(value);
  const commit = () => {
    const next = draft.trim();
    if (next !== value.trim()) onCommit(next);
  };

  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <label htmlFor={inputId} className="text-sm font-medium text-foreground">
        {label}
      </label>
      <Input
        id={inputId}
        type="text"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
          }
        }}
        placeholder={placeholder}
        disabled={disabled}
        spellCheck={false}
        autoComplete="off"
        className="h-9 font-mono text-sm"
      />
    </div>
  );
}
