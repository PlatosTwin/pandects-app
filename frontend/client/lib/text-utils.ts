import { TEXT_TRUNCATION_LENGTH } from "./constants";

/**
 * Utility function to truncate text for display with tooltip info
 */
export const truncateText = (
  text: string,
  maxLength: number = TEXT_TRUNCATION_LENGTH,
) => {
  if (text.length <= maxLength) {
    return { truncated: text, needsTooltip: false };
  }
  return {
    truncated: text.substring(0, maxLength) + "...",
    needsTooltip: true,
  };
};

/**
 * Get the plural label for filter dropdowns
 * Uses hardcoded mappings for consistent, professional labels
 */
export const pluralizeLabel = (label: string): string => {
  const labelMap: Record<string, string> = {
    // Main filters
    "Year": "years",
    "Target": "targets",
    "Acquirer": "acquirers",
    "Clause Type": "clause types",
    "Section Type": "section types",
    
    // Transaction Price filters (numeric ranges)
    "Total": "values",
    "Stock": "values",
    "Cash": "values",
    "Assets": "values",
    
    // Other filters
    "Transaction Consideration": "types",
    "Target Type": "types",
    "Acquirer Type": "types",
    "Target Counsel": "firms",
    "Acquirer Counsel": "firms",
    "Target Industry": "industries",
    "Acquirer Industry": "industries",
    "Deal Status": "statuses",
    "Attitude": "attitudes",
    "Deal Type": "types",
    "Purpose": "purposes",
    
    // Boolean filters
    "Target PE": "values",
    "Acquirer PE": "values",
  };
  
  return labelMap[label] || `${label}s`;
};

/**
 * Format filter option for display
 * Converts underscores to spaces and uses sentence case (first letter capitalized)
 */
export const formatFilterOption = (option: string): string => {
  // Replace underscores with spaces
  let formatted = option.replace(/_/g, " ");
  
  // Sentence case: capitalize first letter only
  formatted = formatted.charAt(0).toUpperCase() + formatted.slice(1).toLowerCase();
  
  return formatted;
};

/**
 * Decode the common XML/HTML entities that show up inside attribute values
 * (e.g. titles pulled out of the agreement XML with regex).
 */
export const decodeXmlEntities = (value: string): string =>
  value.replace(
    /&(#\d+|#x[0-9a-fA-F]+|amp|lt|gt|quot|apos);/g,
    (match, entity: string) => {
      if (entity === "amp") return "&";
      if (entity === "lt") return "<";
      if (entity === "gt") return ">";
      if (entity === "quot") return '"';
      if (entity === "apos") return "'";
      if (entity[0] === "#") {
        const code =
          entity[1] === "x" || entity[1] === "X"
            ? parseInt(entity.slice(2), 16)
            : parseInt(entity.slice(1), 10);
        return Number.isFinite(code) ? String.fromCodePoint(code) : match;
      }
      return match;
    },
  );

/**
 * Pluralize a word with proper handling of common rules
 */
export const pluralize = (word: string): string => {
  if (word.endsWith("y")) return word.slice(0, -1) + "ies";
  if (word.endsWith("s")) return word + "es";
  return word + "s";
};
