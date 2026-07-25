/**
 * Single source of truth for chart colors.
 *
 * Two kinds of values live here:
 * - Theme-token strings (`hsl(var(--...))`) that resolve through CSS variables
 *   and therefore follow any future dark-mode token changes automatically.
 * - Literal hsl series colors that have no design-token equivalent today.
 *   A dark mode that needs different series colors should change them here
 *   (or promote them to CSS variables) — no chart component declares its own.
 */

// ---------------------------------------------------------------------------
// Categorical series colors (literal values; no matching design token exists).
// ---------------------------------------------------------------------------

export const CHART_SERIES_COLORS = {
  blue: "hsl(212 93% 50%)",
  teal: "hsl(170 84% 36%)",
  amber: "hsl(35 92% 52%)",
  red: "hsl(0 84% 60%)",
  cyan: "hsl(196 83% 42%)",
  purple: "hsl(262 83% 58%)",
  green: "hsl(142 71% 45%)",
  orange: "hsl(24 95% 53%)",
  indigo: "hsl(221 83% 53%)",
  coral: "hsl(12 76% 61%)",
  deepCyan: "hsl(187 72% 41%)",
  violet: "hsl(271 81% 56%)",
  yellow: "hsl(43 96% 56%)",
  seaGreen: "hsl(151 55% 42%)",
  skyBlue: "hsl(215 70% 59%)",
} as const;

/** Ordered categorical palette for many-series charts (counsel leaderboards). */
export const CHART_SERIES_PALETTE: readonly string[] = [
  CHART_SERIES_COLORS.blue,
  CHART_SERIES_COLORS.teal,
  CHART_SERIES_COLORS.amber,
  CHART_SERIES_COLORS.red,
  CHART_SERIES_COLORS.cyan,
  CHART_SERIES_COLORS.purple,
  CHART_SERIES_COLORS.green,
  CHART_SERIES_COLORS.orange,
  CHART_SERIES_COLORS.indigo,
  CHART_SERIES_COLORS.coral,
  CHART_SERIES_COLORS.deepCyan,
  CHART_SERIES_COLORS.violet,
  CHART_SERIES_COLORS.yellow,
  CHART_SERIES_COLORS.seaGreen,
  CHART_SERIES_COLORS.skyBlue,
];

/**
 * Seven-color prefix used by industry-composition charts. Kept as a fixed
 * slice so modulo cycling over series indexes keeps today's assignments.
 */
export const INDUSTRY_CHART_PALETTE: readonly string[] =
  CHART_SERIES_PALETTE.slice(0, 7);

/** Neutral gray for "unknown" / "other" catch-all series. */
export const CHART_NEUTRAL_SERIES_COLOR = "hsl(220 9% 60%)";

// ---------------------------------------------------------------------------
// Semantic series assignments.
// ---------------------------------------------------------------------------

/** Public vs. private target split (trends charts, median-band chart). */
export const OWNERSHIP_SERIES_COLORS = {
  public: CHART_SERIES_COLORS.blue,
  private: CHART_SERIES_COLORS.teal,
} as const;

/** Deal-type series on the agreement index overview. */
export const DEAL_TYPE_COLORS: Record<string, string> = {
  merger: CHART_SERIES_COLORS.blue,
  stock_acquisition: CHART_SERIES_COLORS.teal,
  asset_acquisition: CHART_SERIES_COLORS.amber,
  membership_interest_purchase: CHART_SERIES_COLORS.cyan,
  tender_offer: CHART_SERIES_COLORS.red,
  unknown: CHART_NEUTRAL_SERIES_COLOR,
};

/** Alternating fallback pair for deal types outside DEAL_TYPE_COLORS. */
export const DEAL_TYPE_FALLBACK_COLORS = [
  "hsl(226 80% 58%)",
  "hsl(191 82% 45%)",
] as const;

/** Pipeline processing-status series on the agreement index charts. */
export const PROCESSING_STATUS_COLORS = {
  processed: CHART_SERIES_COLORS.green,
  staged: "hsl(38 92% 55%)",
  awaiting: CHART_SERIES_COLORS.red,
  notPaginated: CHART_NEUTRAL_SERIES_COLOR,
} as const;

/** Sector-concentration trend line (Trends & Analyses). */
export const SECTOR_CONCENTRATION_LINE_COLOR = CHART_SERIES_COLORS.coral;

/** Heatmap cell fill: primary series blue at a data-driven opacity. */
export function heatmapCellFill(opacity: number): string {
  return `hsl(212 93% 50% / ${opacity})`;
}

// ---------------------------------------------------------------------------
// Structural chart colors (resolved via theme tokens).
// ---------------------------------------------------------------------------

/** Horizontal gridlines. */
export const CHART_GRID_STROKE = "hsl(var(--border) / 0.4)";
/** Dashed minor-year vertical guides. */
export const YEAR_GRID_MINOR_STROKE = "hsl(var(--border) / 0.18)";
/** Solid major-year vertical guides. */
export const YEAR_GRID_MAJOR_STROKE = "hsl(var(--border) / 0.32)";
/** Reference lines (e.g. the 2020/2021 source-split marker). */
export const CHART_REFERENCE_LINE_STROKE = "hsl(var(--foreground))";
/** In-chart floating label chip (reference-line marker). */
export const CHART_MARKER_FILL = "hsl(var(--background))";
export const CHART_MARKER_STROKE = "hsl(var(--border))";
export const CHART_MARKER_TEXT_FILL = "hsl(var(--muted-foreground))";
