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
