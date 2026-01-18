/**
 * Shared formatting utilities for consistent data display across the application
 */

/**
 * Formats a date string to a localized date-time string
 * @param value - Date string or null
 * @returns Formatted date string or "—" if invalid/null
 */
export function formatDate(value: string | null): string {
  if (!value) return "—";
  const dt = new Date(value);
  return Number.isNaN(dt.getTime()) ? value : dt.toLocaleString();
}

/**
 * Formats a date string to a short date format (e.g., "Jan 15, 2024")
 * @param value - Date string or null
 * @returns Formatted date string or "—" if invalid/null
 */
export function formatDateValue(value?: string | null): string {
  if (!value) return "—";
  try {
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "2-digit",
    }).format(new Date(value));
  } catch {
    return "—";
  }
}

/**
 * Formats a number with locale-specific formatting
 * @param value - Number or null
 * @param options - Intl.NumberFormatOptions
 * @returns Formatted number string or "—" if null/undefined
 */
export function formatNumberValue(
  value?: number | null,
  options?: Intl.NumberFormatOptions,
): string {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 3,
    ...options,
  }).format(value);
}

/**
 * Formats a number as currency (USD)
 * @param value - Number or null
 * @returns Formatted currency string or "—" if null/undefined
 */
export function formatCurrencyValue(value?: number | null): string {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
}

/**
 * Formats a number with locale-specific formatting (simple version)
 * @param value - Number
 * @returns Formatted number string
 */
export function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value);
}

/**
 * Formats an enum value by converting underscores to spaces and capitalizing
 * @param value - Enum string or null
 * @returns Formatted string or "—" if null/undefined
 */
export function formatEnumValue(value?: string | null): string {
  if (!value) return "—";
  return String(value)
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

/**
 * Formats a boolean value
 * @param value - Boolean or null
 * @returns "Yes", "No", or "—"
 */
export function formatBooleanValue(value?: boolean | null): string {
  if (value === null || value === undefined) return "—";
  return value ? "Yes" : "No";
}

/**
 * Formats a text value, returning "—" for null/undefined/empty
 * @param value - String or null
 * @returns String value or "—"
 */
export function formatTextValue(value?: string | null): string {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}
