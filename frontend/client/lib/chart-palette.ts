/**
 * Single source of truth for chart colors.
 *
 * Every value resolves through a CSS variable (`hsl(var(--...))`) defined in
 * `client/global.css` for both `:root` and `.dark`, so charts follow the theme.
 * The categorical palette is capped at 8 series (`--chart-1..8`); a chart
 * that needs more should fold the tail into an "Other" series colored with
 * CHART_NEUTRAL_SERIES_COLOR rather than reuse hues.
 */

// ---------------------------------------------------------------------------
// Categorical series colors.
// ---------------------------------------------------------------------------

const chartVar = (n: number) => `hsl(var(--chart-${n}))`;

/** Maximum distinct categorical series a chart should draw. */
export const CHART_MAX_CATEGORICAL_SERIES = 8;

export const CHART_SERIES_COLORS = {
  blue: chartVar(1),
  teal: chartVar(2),
  amber: chartVar(3),
  red: chartVar(4),
  cyan: chartVar(5),
  purple: chartVar(6),
  green: chartVar(7),
  orange: chartVar(8),
  /** Aliases kept for existing callers; they map onto the 8-series palette. */
  coral: chartVar(4),
} as const;

/** Ordered categorical palette (8 series). */
export const CHART_SERIES_PALETTE: readonly string[] = [1, 2, 3, 4, 5, 6, 7, 8].map(
  chartVar,
);

/**
 * Seven-color prefix used by industry-composition charts. Kept as a fixed
 * slice so modulo cycling over series indexes keeps today's assignments.
 */
export const INDUSTRY_CHART_PALETTE: readonly string[] =
  CHART_SERIES_PALETTE.slice(0, 7);

/** Neutral gray for "unknown" / "other" catch-all series. */
export const CHART_NEUTRAL_SERIES_COLOR = "hsl(var(--chart-neutral))";

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
  CHART_SERIES_COLORS.purple,
  CHART_SERIES_COLORS.orange,
] as const;

/** Pipeline processing-status series on the agreement index charts. */
export const PROCESSING_STATUS_COLORS = {
  processed: CHART_SERIES_COLORS.green,
  staged: CHART_SERIES_COLORS.amber,
  awaiting: CHART_SERIES_COLORS.red,
  notPaginated: CHART_NEUTRAL_SERIES_COLOR,
} as const;

/** Sector-concentration trend line (Trends & Analyses). */
export const SECTOR_CONCENTRATION_LINE_COLOR = CHART_SERIES_COLORS.coral;

/** Heatmap cell fill: primary series blue at a data-driven opacity. */
export function heatmapCellFill(opacity: number): string {
  return `hsl(var(--chart-1) / ${opacity})`;
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
