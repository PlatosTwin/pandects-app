# Frontend Style Guide

Working reference for the design system as implemented. Tokens live in
`client/global.css`, the Tailwind theme in `tailwind.config.ts`, chart colors
in `client/lib/chart-palette.ts`, badges in `client/components/ui/badge.tsx`.

## Typography

Base font is Geist with an Inter Variable fallback (`body` in `global.css`).
`font-mono` is Tailwind's default mono stack — used for IDs, file formats, and
pipeline/code identifiers. Root font size is 16.5px, so rem-based sizes render
slightly larger than Tailwind's nominal px values; the arbitrary `text-[Npx]`
sizes below are exact.

### Standard scale (rem-based)

| Class | Role |
|-------|------|
| `text-5xl` / `text-4xl` | Landing hero only |
| `text-3xl` | `h1` default (semibold, `-0.02em` tracking, set globally) |
| `text-2xl` | Section headers (`SectionHeader.tsx`) |
| `text-xl` | `h2` default (semibold, `-0.015em`); card titles |
| `text-lg` | `h3` default (semibold, `-0.01em`); result titles |
| `text-base` | Emphasized body, mobile result titles |
| `text-sm` | Default body/UI text (most common size) |
| `text-xs` | Secondary UI text: badges, table meta, captions, helper text |

Headings get their size and tracking from the `@layer base` rules in
`global.css` — do not re-specify `text-3xl font-semibold` on an `h1` unless
overriding.

### Micro-label system (arbitrary px sizes)

An intentional tier below `text-xs` for dense data UI. Use these exact sizes;
do not invent new arbitrary values.

| Class | Role | Conventions |
|-------|------|-------------|
| `text-[11px]` | Field labels ("Target", "Acquirer"), eval-table headers, micro chips (`Badge` `chip` variant) | Usually `font-semibold uppercase tracking-wide text-muted-foreground` |
| `text-[10px]` | Smallest tier: BETA pill, reader-toolbar labels, mono file-format chips, tiny count pills | Same uppercase treatment, or `font-mono font-normal` for format chips |
| `text-[13px]` | Taxonomy L1 mono IDs only — a deliberate per-level step (L1 `13px` → L2/L3 `11px`) | `font-mono text-foreground/80` |
| `text-[9px]` | `.tooltip-help-trigger-compact` glyph only | Defined in `global.css`, not used inline |

Rule of thumb: `text-xs` is the floor for readable sentences; the px tiers are
for labels, chips, and IDs where density matters more than reading comfort.

Letter-spacing: uppercase micro labels use `tracking-wide`. The bespoke
`tracking-[0.14em]` (BETA pill), `tracking-[0.2em]` (Landing tagline), and
`tracking-[0.24em]` (Login divider) are one-off brand accents — don't reuse
them for data UI.

## Color

### Semantic tokens

All colors are HSL CSS variables in `client/global.css` (`:root` for light,
`.dark` for dark; dark mode is class-based, toggled from the nav). Tailwind
maps them in `tailwind.config.ts`. Use the semantic classes — never raw hexes
or `hsl(...)` literals in components.

| Token | Role |
|-------|------|
| `background` / `foreground` | Surfaces and primary text |
| `cream` | App/page background (`body` uses `bg-cream`) — cool near-white in light, near-black in dark |
| `card`, `popover` | Elevated surfaces |
| `primary` | Brand cobalt; actions, active nav, links, tinted counts (`bg-primary/10 text-primary`) |
| `secondary` | Soft fills; `secondary-foreground` also powers `.prose-copy` body-copy color |
| `muted` / `muted-foreground` | Subdued fills and secondary text |
| `accent` | Hover fills (`hover:bg-accent/60`) |
| `destructive` | Errors, destructive actions |
| `border` / `input` / `ring` | Separators (`border-border`, subtle: `border-border/50`), form borders, focus rings |

Opacity modifiers stay on the small approved set already in use (`/5`, `/10`,
`/20`, `/25`, `/30`, `/40`, `/50`, `/60`, `/70`, `/80`, `/95`); text emphasis
steps use `/80`.

### Status and accent conventions

- **Emerald** = positive status: verified badges and success chips use
  `bg-emerald-500/10 text-emerald-700 ring-emerald-500/20 dark:text-emerald-300`
  (the Badge `success` variant); pipeline-progress gradients and eval highlight
  panels use emerald tints with explicit `dark:` counterparts.
- **Amber** = caution / markup highlight: unverified-data callouts
  (`TaxClauseResultsList`), before/after markup highlighting in Sources &
  Methods (`bg-amber-100/80 ... dark:bg-amber-500/20`).
- **Red** goes through the `destructive` token, not Tailwind red classes — with one
  systematic exception: `--destructive` doubles as a button background and is too dark
  to read as text on dark surfaces, so destructive *text* pairs it with a
  `dark:text-red-400` override (see `ui/alert.tsx`). Follow that pattern; don't
  "fix" it back to the bare token.
- Fixed-color exceptions: the Google sign-in button (`Login.tsx`) keeps literal
  slate/white brand colors in both themes. Anything else hardcoded needs a
  `dark:` variant and a reason.

### Charts

All chart colors import from `client/lib/chart-palette.ts` — no chart component
declares its own. The module has three groups:

1. **Categorical series** (`CHART_SERIES_COLORS`, `CHART_SERIES_PALETTE`,
   `INDUSTRY_CHART_PALETTE`): literal HSL values with no token equivalent;
   change them there, never inline.
2. **Semantic assignments** (`OWNERSHIP_SERIES_COLORS`, `DEAL_TYPE_COLORS`,
   `PROCESSING_STATUS_COLORS`, `SECTOR_CONCENTRATION_LINE_COLOR`,
   `CHART_NEUTRAL_SERIES_COLOR` for unknown/other).
3. **Structural strokes** (grid, guides, reference lines, marker chips):
   `hsl(var(--token) / alpha)` strings that follow theme changes automatically.

## Badges and pills

`client/components/ui/badge.tsx` renders a `span` (valid inside buttons,
headings, prose) and exports `Badge` plus `badgeVariants` for interactive
elements that need pill styling on a `button` or `a`.

| Variant | Look | Use for |
|---------|------|---------|
| `default` | Solid primary | Primary emphasis (taxonomy level chips override its bg) |
| `secondary` | Soft secondary fill | Generic counts/labels ("3 sections") |
| `destructive` | Solid destructive | Error states |
| `outline` | Bordered, foreground text | Years, deal metadata on cards/tables |
| `success` | Emerald tint + ring | Verified/positive status chips |
| `metadata` | On-background + `ring-border`, muted text | Neutral metadata pills (clause type, year in modal) |
| `muted` | `bg-muted`, muted text | Soft info pills (article/section titles, enum values) |
| `chip` | Bordered `bg-muted/40`, 11px | Micro-label chips (eval labels, doc/pipeline tags); add `font-mono font-normal text-[10px]` for file-format chips |
| `count` | `bg-primary/10 text-primary` | Primary-tinted count pills (taxonomy group/type counts) |

Interactive pills stay real `button`/`a` elements and compose
`cn(badgeVariants({ variant }), ...)` — see the verified/clause-type triggers
in `SearchResultsTable.tsx` and the anchor chips in `SourcesMethods.tsx`.

Deliberately **not** badges: the nav BETA pill and Landing tagline (bespoke
tracking/brand accents), `MetaPill`/`HeaderFactChip` label+value fact chips,
the SEC-filing link pill, filter remove-pills, chart-legend pills
(`Leaderboards`, `TrendsAnalyses`), segmented controls, step-number circles,
and status dots.
