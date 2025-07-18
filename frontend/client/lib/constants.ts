/**
 * Application constants
 */

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
