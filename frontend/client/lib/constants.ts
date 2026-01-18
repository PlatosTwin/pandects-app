/**
 * Application constants
 */

// Beta notice
export const BETA_SAMPLE_AGREEMENTS_COUNT = 45;

// Pagination
export const DEFAULT_PAGE_SIZE = 25;
export const DEFAULT_PAGE = 1;
export const LARGE_PAGE_SIZE_FOR_CSV = 10000;

// UI Constants
export const DROPDOWN_ANIMATION_DELAY = 100;
export const SIDEBAR_ANIMATION_DELAY = 320;
export const TEXT_TRUNCATION_LENGTH = 40;

// Search Configuration
export const SEARCH_DEBOUNCE_DELAY = 300;

// Years range for filtering (could be dynamic in the future)
export const AVAILABLE_YEARS = [
  "2020",
  "2019",
  "2018",
  "2017",
  "2016",
  "2015",
  "2014",
  "2013",
  "2012",
  "2011",
  "2010",
  "2009",
  "2008",
  "2007",
  "2006",
  "2005",
  "2004",
  "2003",
  "2002",
  "2001",
  "2000",
];

// Transaction filter options
export const TRANSACTION_SIZE_OPTIONS = [
  "100M - 250M",
  "250M - 500M",
  "500M - 750M",
  "750M - 1B",
  "1B - 5B",
  "5B - 10B",
  "10B - 20B",
  "20B+",
];

export const TRANSACTION_TYPE_OPTIONS = ["Strategic", "Financial"];

export const CONSIDERATION_TYPE_OPTIONS = ["All stock", "All cash", "Mixed"];

export const TARGET_TYPE_OPTIONS = ["Public", "Private"];

// Text truncation constants
export const DEFAULT_TRUNCATION_LENGTH = 75;
export const LONG_TRUNCATION_LENGTH = 120;

// Breakpoint constants (matching Tailwind defaults)
export const BREAKPOINT_SM = 640; // sm
export const BREAKPOINT_MD = 768; // md
export const BREAKPOINT_LG = 1024; // lg
export const BREAKPOINT_XL = 1280; // xl
export const BREAKPOINT_2XL = 1536; // 2xl

// Icon size constants
export const ICON_SIZE_SM = "h-4 w-4"; // Inline with text
export const ICON_SIZE_MD = "h-5 w-5"; // Buttons
export const ICON_SIZE_LG = "h-6 w-6"; // Headings

// Spacing scale constants (gap/padding)
export const SPACING_TIGHT = "gap-2"; // 0.5rem / 8px
export const SPACING_DEFAULT = "gap-4"; // 1rem / 16px
export const SPACING_COMFORTABLE = "gap-6"; // 1.5rem / 24px
export const SPACING_SPACIOUS = "gap-8"; // 2rem / 32px
