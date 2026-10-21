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

Eyebrows (uppercase labels above titles, stats and control groups) use the
`Eyebrow` component (`components/ui/eyebrow.tsx`, or `eyebrowClassName` on
elements you can't wrap): `text-xs font-medium uppercase tracking-wider
text-muted-foreground`. One size, one tracking. Don't hand-roll
`tracking-[...]` values; the nav BETA pill and Landing tagline are the only
remaining bespoke brand accents.

Page titles come from `PageHeader` (through `PageShell`): the h1 uses
`clamp(1.625rem, 1.25rem + 1.25vw, 2rem)` so it never wraps at desktop width.

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

### Status tokens

Status color always goes through a semantic token, never Tailwind
`emerald-*`/`amber-*`/`red-*`/`blue-*` classes. Each token passes WCAG AA as
text on `card` in both themes, and pairs with its `-foreground` for solid fills.
Soft fills use the `/5`, `/10`, `/15`, `/20` opacity steps.

| Token | Meaning | Text | Soft fill | Solid |
|-------|---------|------|-----------|-------|
| `success` | Verified, passed, complete | `text-success` | `bg-success/10` | `bg-success text-success-foreground` |
| `warning` | Caution, unverified, markup highlight | `text-warning` | `bg-warning/10` | `bg-warning text-warning-foreground` |
| `info` | Neutral notice | `text-info` | `bg-info/10` | `bg-info text-info-foreground` |
| `destructive` | Errors, destructive actions | `text-destructive` | `bg-destructive/10` | `bg-destructive text-destructive-foreground` |

- Dark `--destructive` is a lighter red (5.6:1 as text on card) and its
  `-foreground` is near-black (5.6:1), the same way dark `--primary` works. No
  `dark:text-red-400` overrides: the bare token is correct in both themes.
- Use the Badge (`success`/`warning`/`info`/`danger`) and Alert
  (`destructive`/`success`/`warning`/`info`) variants before hand-rolling.
- Categorical palettes are not status: the favorites tag color picker
  (`favorites/TagPill.tsx`) keeps its Tailwind hues on purpose.
- Error dialogs (`ErrorModal`): red on the icon only; the Close button is
  `variant="outline"`.

### Radius

One closed scale derived from `--radius` (0.375rem): `rounded-sm` (2px) <
`rounded-md` (4px) < `rounded-lg` (6px: buttons, inputs, menus) < `rounded-xl`
(10px: cards, panels) < `rounded-2xl` (14px: rare large surfaces).
`rounded-full` is for pills, avatars, dots and icon-only circular controls —
never for text buttons. Buttons use the `rounded-lg` default; don't override
it per page.

### Charts

All chart colors import from `client/lib/chart-palette.ts` — no chart component
declares its own. The module has three groups:

1. **Categorical series** (`CHART_SERIES_COLORS`, `CHART_SERIES_PALETTE`,
   `INDUSTRY_CHART_PALETTE`): `hsl(var(--chart-1..8))`, defined in
   `global.css` with dark variants (Tailwind: `bg-chart-1`...). Max 8 series
   (`CHART_MAX_CATEGORICAL_SERIES`); fold the tail into an "Other" series
   colored `--chart-neutral` instead of cycling hues.
   Every light-mode series color, `--chart-neutral` included, keeps at least
   3:1 contrast against `--card` and `--background` (WCAG non-text
   contrast); change `critical.css` in step with `global.css`.
   Heatmap cells (`heatmapCellFill`) use `--heatmap`, not `--chart-1`: its
   dark value stays darker so foreground text keeps 4.5:1 at every opacity.
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
| `success` / `warning` / `info` / `danger` | Status-token tint + ring | Status chips (verified, unverified, notice, failed) |
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

## Page frame

Every page renders inside `PageShell` (`components/PageShell.tsx`). The outer
frame is the `app-frame` class (`global.css`: `max-w-screen-2xl` plus the
`px-4 sm:px-6 lg:px-8` gutter), the same class the navigation bar, footer
and the reader use, so the title always lines up with the logo. Change
the width there only; `lib/design-system.spec.ts` fails on a copied
`max-w-7xl`. Search is the exception: a full-bleed workspace with the
filter rail flush against the screen edge, like most search UIs. Pick one `layout`; it sets the page's
single reading measure. Never add per-paragraph `max-w-NNch` caps inside it.

| `layout` | Content width | Use for |
|----------|---------------|---------|
| `docs` | 720px; with `toc`, a sticky 240px TOC column on the left (lg+) | About, legal, Sources & Methods, XML schema, support |
| `data` | The full frame | Tables, charts, dashboards, account |
| `form` | `max-w-md`, left-aligned under the h1 | Sign in, sign up, password reset, consent |

Props: `layout`, `eyebrow?`, `title?`, `description?`, `actions?`, `toc?`
(docs only), `className?`, `contentClassName?`. The `size` and `subtitle`
props are deprecated; `size` without `layout` still renders the old centered
frame for pages not migrated yet (Taxonomy, AgreementIndex, the Zitadel
callback, the full-bleed Favorites view).

## Elevation and motion

Depth comes from one scale, set per theme in `global.css` and exposed as
Tailwind shadows. Never use `shadow-sm/md/lg/xl`; the design-system spec fails
on them, and on any bordered, rounded `bg-card` surface with no shadow.

| Class | Use for |
|-------|---------|
| `shadow-raised` | Resting surfaces: cards, panels, result rows, chips, outline and primary buttons. `Card` has it built in. |
| `shadow-lifted` | Hovered or selected surfaces; `Card variant="elevated"`. |
| `shadow-overlay` | Popovers, menus, selects, tooltips, dialogs, sheets, floating cards. |

- A card that is itself one link or button (starter tiles, linked
  `FactCard`s) also gets `surface-interactive`: on hover the border picks up
  primary, the shadow lifts and the card rises 1px; it settles on press.
  Motion is skipped under `prefers-reduced-motion`. Cards that only contain
  links or are selectable (result rows, matched sections) don't rise; they
  take `hover:border-primary/30` (and `hover:shadow-lifted`) instead.
- New content fades up with `motion-safe:animate-enter` (220ms): result cards,
  an expanded deal row, the landing card. Use it for content that arrives,
  not for every re-render.
- Primary buttons press in (`active:scale-[0.98]`); list-row chevrons nudge
  right on hover (`motion-safe:group-hover:translate-x-0.5`).
- An open table row and its details share a 3px primary rail on the left.

## Links, overlays and copy

- **Reference links.** From a working surface (search, the reader, a terms
  prompt) a link to an in-app reference page (Taxonomy, Sources & Methods,
  XML schema, legal pages) uses `ReferenceLink`, which opens a new tab so the
  user keeps their results. `lib/reference-links.spec.ts` enforces this.
  Other sites use `ExternalLink`.
- **Scroll lock.** Radix menus, selects, dialogs and sheets lock page scroll.
  For menus and selects, the `html body[data-scroll-locked]` rule in
  `global.css` keeps the scrollbar and body box unchanged, so nothing shifts
  sideways; dialogs and sheets keep the full lock. Don't remove it;
  `lib/scroll-lock.spec.ts` checks it against the library.
- **Tooltip copy** describes what the control does, in the third person:
  "Opens the Taxonomy page in a new tab", not "open the Taxonomy page".
  Button labels stay imperative ("Open agreement").
- **Filter chips** group values of the same filter into one chip with a "+N"
  list (`groupChips` in `SearchActiveFilters.tsx`); the chip's X clears the
  whole filter.
- **Theme toggle** shows a sun or moon for the theme on screen; "System" is a
  menu choice, not the trigger icon.

## Shared primitives

| Component | File | Props |
|-----------|------|-------|
| `PageHeader` | `components/PageHeader.tsx` | `eyebrow?`, `title`, `description?`, `actions?`, `titleId?` |
| `Eyebrow` | `components/ui/eyebrow.tsx` | `asChild?` + HTML attrs; also `eyebrowClassName` |
| `SectionHeader` | `components/SectionHeader.tsx` | `title`, `id`, `description?`, `actions?`, `number?` (opt-in chip; avoid) |
| `FactCard` | `components/FactCard.tsx` | `icon?`, `eyebrow?`, `title`, `description?`, `href?` (router path, `http(s)`, or `mailto:`), `titleAs?` |
| `ChartPanel` | `components/ChartPanel.tsx` | `title`, `description?`, `controls?`, `children` (chart), `footnote?`, `mobileExpand?` (`true` or a render fn for the modal), `titleAs?`, `id?` |
| `SegmentedControl` | `components/ui/segmented-control.tsx` | `value`, `onValueChange`, `options: {value, label, disabled?}[]`, `aria-label`, `size?: "sm" \| "default"` |
| `ErrorState` | `components/StateMessage.tsx` | `error?` (`{status?, code?, message?}`, `Error`, or string), `title?`, `body?`, `onRetry?`, `action?`, `signInHref?`, `variant?: "card" \| "inline"` |
| `EmptyState` | `components/StateMessage.tsx` | `icon?`, `title`, `body?`, `action?`, `variant?` |
| `describeError` | `components/StateMessage.tsx` | Maps 401/403 → sign in, 404, 429 → slow down, network → retry, 5xx |
| `StatusPage` | `components/StatusPage.tsx` | `code?` or `icon?`, `title`, `description?`, `actions?`, `children?`, `titleAs?`, `role?` |
| `CopyButton` | `components/ui/copy-button.tsx` | `value` (string or fn), `label?` (a11y name), `showLabel?`, `idleText?`, `copiedMessage?`, `failedMessage?`, `onCopied?`, Button props; 32px target |
| `useCopyToClipboard` | `components/ui/copy-button.tsx` | `(resetMs?) => { state, copy(text) }` |
| `CodeBlock` | `components/ui/code-block.tsx` | `code`, `label?`, `copyLabel?`, `copyDisabled?`, `onCopied?`, `wrap?` |
| `DocsTableOfContents` | `components/DocsTableOfContents.tsx` | `items: {id, label, level?: 0 \| 1}[]`, `title?`; scroll-spy (`rootMargin: -20% 0px -70% 0px`) and a mobile "On this page" disclosure. Pass it as `PageShell` `toc`. Not the agreement reader's TOC. |
| `ExternalLink` | `components/ui/external-link.tsx` | `href`, `hideIcon?` + anchor attrs; adds the glyph, `target`, `rel`, and an sr-only "(opens in a new tab)" |

Surfaces: `backdrop-blur` is for sticky bars and overlays only, never static
cards. Cards are opaque `bg-card`.
