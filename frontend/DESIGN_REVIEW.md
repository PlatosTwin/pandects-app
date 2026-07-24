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
- **Section headers** — shared `SectionHeader.tsx` (numbered chip +
  `text-2xl font-semibold`).
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

## Remaining (larger efforts, not for a polish pass)

- **Typography scale documentation** — `text-[11px]` / `text-[10px]` micro-label
  sizes are a consistent de facto system (~50 uses) but undocumented; a proper
  audit would name them (and decide whether `text-[13px]` mono IDs in Taxonomy
  stay a deliberate per-level step).
- **Badge unification** — two internally consistent systems coexist: the
  `Badge` component (`border`, `px-2.5`) and inline `rounded-full ... ring-1`
  pill spans (`SearchResultsTable.tsx`, `AgreementModal.tsx`). Unifying them
  means a `Badge` variant + sweep.
- **Color-usage documentation** — primary/muted/accent role conventions are
  followed in practice but written down nowhere.
