/**
 * Display form for an industry value from agreement data, using the NAICS
 * label the API returns next to it (`target_industry_label`,
 * `acquirer_industry_label`, or `industry_labels[code]` from filter options).
 *
 * - Code with a label → "Label (code)". The code stays in the display because
 *   sector labels are not unique (31–33 are all "Manufacturing").
 * - No label → the value itself: a non-numeric value is already readable, and
 *   an unresolvable code is still more useful than nothing.
 * - Empty → null.
 */
export function formatIndustry(
  value: string | null | undefined,
  label: string | null | undefined,
): string | null {
  const trimmed = value?.trim();
  if (!trimmed) return null;
  const name = label?.trim();
  if (!name || name === trimmed) return trimmed;
  return `${name} (${trimmed})`;
}
