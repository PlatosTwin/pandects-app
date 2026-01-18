/**
 * Design tokens for consistent UI styling
 * These constants should be used throughout the application for visual consistency
 */

// Border radius scale
export const BORDER_RADIUS = {
  sm: "rounded-sm", // 2px - small elements
  md: "rounded-md", // 6px - buttons, inputs, cards (default)
  lg: "rounded-lg", // 8px - containers
  xl: "rounded-xl", // 12px - large containers
  "2xl": "rounded-2xl", // 16px - hero sections
  full: "rounded-full", // 9999px - pills, badges, avatars
} as const;

// Shadow scale
export const SHADOW = {
  sm: "shadow-sm", // Cards
  md: "shadow-md", // Dropdowns
  lg: "shadow-lg", // Modals
  xl: "shadow-xl", // Overlays
  none: "shadow-none",
} as const;

// Focus ring (standardized)
export const FOCUS_RING = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";

// Backdrop blur (with fallback)
export const BACKDROP_BLUR = "backdrop-blur supports-[backdrop-filter]:backdrop-blur";

// Icon sizes (use with h-* w-* classes)
export const ICON_SIZE = {
  xs: "h-3 w-3", // Extra small (12px)
  sm: "h-4 w-4", // Small - inline with text (16px)
  md: "h-5 w-5", // Medium - buttons (20px)
  lg: "h-6 w-6", // Large - headings (24px)
  xl: "h-8 w-8", // Extra large (32px)
} as const;

// Spacing scale (gap/padding)
export const SPACING = {
  tight: "gap-2", // 0.5rem / 8px
  default: "gap-4", // 1rem / 16px
  comfortable: "gap-6", // 1.5rem / 24px
  spacious: "gap-8", // 2rem / 32px
} as const;

// Typography scale
export const TYPOGRAPHY = {
  xs: "text-xs", // 12px
  sm: "text-sm", // 14px
  base: "text-base", // 16px
  lg: "text-lg", // 18px
  xl: "text-xl", // 20px
  "2xl": "text-2xl", // 24px
  "3xl": "text-3xl", // 30px
  "4xl": "text-4xl", // 36px
} as const;
