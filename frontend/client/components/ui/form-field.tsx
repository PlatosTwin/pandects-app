import { cn } from "@/lib/utils";
import { Label } from "@/components/ui/label";
import type { ReactNode } from "react";

interface FormFieldProps {
  label: string;
  htmlFor?: string;
  required?: boolean;
  error?: string;
  helpText?: string;
  children: ReactNode;
  className?: string;
}

/**
 * FormField wrapper component for consistent form styling
 * Provides consistent spacing, label positioning, and error/help text display
 */
export function FormField({
  label,
  htmlFor,
  required = false,
  error,
  helpText,
  children,
  className,
}: FormFieldProps) {
  const fieldId = htmlFor || `field-${label.toLowerCase().replace(/\s+/g, "-")}`;

  return (
    <div className={cn("space-y-2", className)}>
      <Label htmlFor={fieldId} className={required ? "after:content-['*'] after:ml-0.5 after:text-destructive" : ""}>
        {label}
      </Label>
      {children}
      {error && (
        <p className="text-sm text-destructive" role="alert" aria-live="polite">
          {error}
        </p>
      )}
      {helpText && !error && (
        <p className="text-sm text-muted-foreground">{helpText}</p>
      )}
    </div>
  );
}
