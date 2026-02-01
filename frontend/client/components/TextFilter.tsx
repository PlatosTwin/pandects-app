import { useId } from "react";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";

interface TextFilterProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

export function TextFilter({
  label,
  value,
  onChange,
  placeholder,
  className,
  disabled = false,
}: TextFilterProps) {
  const labelId = useId();
  const inputId = useId();

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <label
        id={labelId}
        htmlFor={inputId}
        className="text-xs font-normal text-muted-foreground tracking-[0.15px]"
      >
        {label}
      </label>
      <Input
        id={inputId}
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className="h-10"
        aria-labelledby={labelId}
      />
    </div>
  );
}
