# Frontend Design Review — Status

Reconciliation of the original thematic design audit against the current code
(post UI-polish pass, July 2026). The original document predated the polish
pass; most of its recommendations have since been implemented or made obsolete
by refactors. Line numbers below refer to current code.

## Applied (this pass and prior)

- **Border opacity** — standardized to `border-border` (primary separators) and
  `border-border/50` (subtle, e.g. the Card `subtle` variant). The last stray
  `border-border/70` (favorites `TagsManager.tsx`) was aligned with its
  `FilterBar.tsx` sibling.
- **Card system** — `card.tsx` has `default` / `compact` / `elevated` / `subtle`
  variants with consistent padding, border, and shadow, plus
  `transition-shadow duration-200` on the base.
- **Buttons** — `button.tsx` base includes `transition-all duration-200`,
  `active:scale-[0.98]`, `[&_svg]:size-4`, and focus-visible rings. No ghost
  buttons remain on primary actions.
- **Tables** — `table.tsx`: row hover `bg-muted/30`, sortable-header hover
  `bg-muted/40`, footer `bg-muted/50`, `border-b border-border` rows.
- **Search results hierarchy** — result cards use
  `shadow-sm transition-all hover:shadow-md`; the selected card gets
  `border-primary/40 border-l-primary bg-primary/5 shadow-md`
  (`SearchResultsTable.tsx`).
- **Navigation active state** — desktop links use
  `border-l-2 border-primary bg-primary/10 font-medium text-primary` with
  `aria-current="page"`; hover is the more subtle `hover:bg-accent/60`
  (`NavigationDesktopMenus.tsx`, `NavigationMobileMenu.tsx`).
- **Section headers** — shared `SectionHeader.tsx`; the numbered chip is
  opt-in (`number?`) and no longer used on About/Feedback/Support.
- **Loading states** — skeleton loaders are used across pages
  (`ui/skeleton.tsx`); empty states have icon + hierarchy treatment
  (`search/SearchResultsPanel.tsx`).
- **Spacing** — no off-scale arbitrary spacing values remain; gaps follow the
  Tailwind scale.
- **Chart colors** — centralized in `client/lib/chart-palette.ts`.
- **Opacity one-offs** — `text-foreground/78` (Taxonomy) normalized to `/80`.

## Obsolete / superseded

- Page-level cohesion notes for Search, Account, and Signup — those pages were
  refactored or rebuilt (Search → mode adapters + `ResultsToolbar`; Account
  rebuilt; signup form temporarily replaced by a disabled notice), so the
  original observations no longer map to the code.
- A separate design-tokens file — Tailwind semantic tokens (`border`, `muted`,
  `card`, `primary/N`) already fill this role; the chart palette is the one
  domain that needed extraction and got it.

## Applied (documentation pass)

- **Typography scale, color usage, badge variants** — documented in
  `STYLEGUIDE.md` (the `text-[13px]` Taxonomy mono IDs stay a deliberate
  per-level step).
- **Badge unification** — `badge.tsx` gained `success` / `metadata` / `muted` /
  `chip` / `count` variants; hand-rolled pill spans were swapped for `Badge`
  (or `badgeVariants` on interactive `button`/`a` pills) across
  `SearchResultsTable`, `TransactionResultsList`, `AgreementModal`,
  `AgreementIndex`, `NerEvalMetrics`, `eval-metrics/shared`, `SourcesMethods`,
  `Taxonomy`, and the favorites-area count pills (`ProjectSidebar`,
  `FilterBar`, `TagsManager` → `variant="muted"` with `px-1.5 text-[10px]`
  size overrides). Deliberate non-badges are listed in `STYLEGUIDE.md`.

## Applied (design-system foundations, October 2026)

- **Status tokens** — `--success` / `--warning` / `--info` (+ `-foreground`)
  in both themes; raw emerald/amber/red status classes replaced in
  SourcesMethods, eval metrics, badges, alerts, toast, Account and ErrorModal.
  Dark `--destructive` now passes AA as text (5.6:1 on card) with a dark
  foreground for solid fills; the `dark:text-red-400` workarounds are gone.
- **Radius** — closed scale (`xl`/`2xl` derived from `--radius`); text buttons
  no longer override to `rounded-full`.
- **Chart palette** — `--chart-1..8` + `--chart-neutral` CSS variables with
  dark variants; 8-series cap. Light-mode teal, amber, green, orange and
  neutral were darkened to reach 3:1 on `card`/`background` (they measured
  2.25–2.97:1). Confusion-matrix cells scale alpha by the
  cell's share of its row.
- **Page frame** — `PageShell layout="docs|data|form"` with `PageHeader`;
  titles align with the nav logo. Migrated: About, Feedback, Support, legal
  pages, BulkData, XmlSchema, SourcesMethods, Leaderboards, TrendsAnalyses,
  Account, auth pages, NotFound, Favorites (loading state).
- **Primitives** — Eyebrow, FactCard, ChartPanel, SegmentedControl,
  ErrorState/EmptyState, StatusPage, CodeBlock/CopyButton,
  DocsTableOfContents, ExternalLink (see `STYLEGUIDE.md`). `radio-group`
  removed (unused).

## Open

- Raw status colors remain in files owned elsewhere: `TaxClauseCompare`,
  `XMLRenderer`, `TaxClauseResultsList`, `AgreementReader`, `StarButton`,
  `TurnstileWidget`, `SearchResultsTable`, `SaveConfirmationModal`,
  `AgreementModal` (several still carry `dark:text-red-400`, now redundant).
- `backdrop-blur` on static surfaces: `Footer`, `Landing` hero card,
  `Favorites` filter bar, `search/SearchResultsPanel`, `search/SearchActionsBar`.
- Hand-rolled clipboard code left in `SearchResultsTable` and `Taxonomy`; use
  `CopyButton` / `useCopyToClipboard`.
- Leaderboards charts draw 15 series against the 8-color palette; fold ranks
  9+ into "Other".
- Pages still on the deprecated `size` prop: Taxonomy, AgreementIndex,
  AuthZitadelCallback, full-bleed Favorites.
