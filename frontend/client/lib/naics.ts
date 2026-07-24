/**
 * NAICS reference data: the response shape of GET /v1/naics plus helpers for
 * resolving raw industry codes from agreement data (e.g. "62", "325") to
 * sector/subsector labels.
 */

export interface NaicsSubSector {
  sub_sector_code: string;
  sub_sector_desc: string;
}

export interface NaicsSector {
  sector_code: string;
  sector_desc: string;
  sector_group: string;
  super_sector: string;
  sub_sectors: NaicsSubSector[];
}

export interface NaicsResponse {
  sectors: NaicsSector[];
}

export type NaicsLabelByCode = Record<string, string>;

export function buildNaicsLabelIndex(
  response: NaicsResponse,
): NaicsLabelByCode {
  const index: NaicsLabelByCode = {};
  for (const sector of response.sectors) {
    index[sector.sector_code] = sector.sector_desc;
    for (const subSector of sector.sub_sectors) {
      index[subSector.sub_sector_code] = subSector.sub_sector_desc;
    }
  }
  return index;
}

/**
 * Resolve a numeric NAICS code to its label. Codes without an exact entry
 * (e.g. 4-digit industry groups) roll up to their 2-digit sector label.
 * Returns null when the code cannot be resolved.
 */
export function lookupNaicsLabel(
  labelByCode: NaicsLabelByCode,
  code: string,
): string | null {
  const exact = labelByCode[code];
  if (exact) return exact;
  if (code.length > 2) {
    return labelByCode[code.slice(0, 2)] ?? null;
  }
  return null;
}

/**
 * Display form for an industry value from agreement data.
 *
 * - Numeric NAICS code with a known label → "Label (code)". The code stays in
 *   the display because sector labels are not unique (31–33 are all
 *   "Manufacturing").
 * - Numeric code without a label (lookup miss or NAICS fetch failure) → null:
 *   never surface a bare numeric code.
 * - Non-numeric value → returned as-is (already human-readable).
 */
export function formatNaicsIndustry(
  labelByCode: NaicsLabelByCode,
  value: string | null | undefined,
): string | null {
  const trimmed = value?.trim();
  if (!trimmed) return null;
  if (!/^\d+$/.test(trimmed)) return trimmed;
  const label = lookupNaicsLabel(labelByCode, trimmed);
  return label ? `${label} (${trimmed})` : null;
}
