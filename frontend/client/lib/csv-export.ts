import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { toast } from "@/hooks/use-toast";

/** Page size requested while exporting; the API clamps it to the caller's tier max. */
export const CSV_EXPORT_PAGE_SIZE = 100;
/** Upper bound on rows in one export, so "all results" never fans out unbounded. */
export const CSV_EXPORT_MAX_ROWS = 2000;
/** Upper bound on requests in one export (anonymous pages are only 10 rows). */
export const CSV_EXPORT_MAX_PAGES = 20;

interface PagedResponse<TRow> {
  results: TRow[];
  total_count: number;
  total_count_is_approximate?: boolean;
  has_next: boolean;
}

export interface CsvExportRows<TRow> {
  rows: TRow[];
  totalCount: number;
  totalCountIsApproximate: boolean;
  truncated: boolean;
}

/**
 * Page through a search endpoint for CSV export.
 *
 * `query` is the committed (submitted) search query string; its page and
 * page_size are replaced on each request. Stops at the last page or at the
 * row/page caps, and reports whether the export is partial.
 */
export async function fetchAllPagesForCsv<TRow>(
  endpoint: string,
  query: string,
): Promise<CsvExportRows<TRow>> {
  const rows: TRow[] = [];
  let totalCount = 0;
  let totalCountIsApproximate = false;
  let hasNext = true;
  let page = 1;

  while (hasNext && page <= CSV_EXPORT_MAX_PAGES && rows.length < CSV_EXPORT_MAX_ROWS) {
    const params = new URLSearchParams(query);
    params.set("page", String(page));
    params.set("page_size", String(CSV_EXPORT_PAGE_SIZE));
    const res = await authFetch(apiUrl(`${endpoint}?${params.toString()}`));
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    const body = (await res.json()) as PagedResponse<TRow>;
    if (page === 1) {
      totalCount = body.total_count;
      totalCountIsApproximate = body.total_count_is_approximate ?? false;
    }
    rows.push(...body.results);
    hasNext = body.has_next && body.results.length > 0;
    page += 1;
  }

  const capped = rows.slice(0, CSV_EXPORT_MAX_ROWS);
  return {
    rows: capped,
    totalCount,
    totalCountIsApproximate,
    truncated: hasNext || capped.length < rows.length,
  };
}

let exportInFlight = false;

/**
 * Run a full CSV fetch with a "preparing" toast, at most one at a time.
 * Returns null when another export is already running.
 */
export async function exportAllPagesForCsv<TRow>(
  endpoint: string,
  query: string,
): Promise<CsvExportRows<TRow> | null> {
  if (exportInFlight) return null;
  exportInFlight = true;
  const progress = toast({ title: "Preparing CSV…", description: "Fetching results to export." });
  try {
    return await fetchAllPagesForCsv<TRow>(endpoint, query);
  } finally {
    progress.dismiss();
    exportInFlight = false;
  }
}

export const csvEscape = (value: string | number | null | undefined): string => {
  if (value === null || value === undefined || value === "") return "";
  return `"${String(value).replace(/"/g, '""')}"`;
};

/** Build the CSV text, trigger a browser download, and revoke the object URL. */
export function downloadCsvFile(
  filenameStem: string,
  headers: string[],
  lines: string[],
): void {
  const csv = [headers.join(","), ...lines].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  const objectUrl = URL.createObjectURL(blob);
  link.href = objectUrl;
  link.download = `${filenameStem}_${new Date().toISOString().split("T")[0]}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
}

/** Tell the user when an export holds fewer rows than the search matched. */
export function notifyCsvTruncated(
  exported: number,
  total: number,
  approximate: boolean,
  isAnonymous: boolean,
): void {
  const totalLabel = `${approximate ? "about " : ""}${total.toLocaleString()}`;
  const nextStep = isAnonymous
    ? `Sign in to export up to ${CSV_EXPORT_MAX_ROWS.toLocaleString()} rows.`
    : "Narrow the filters to export the rest.";
  toast({
    title: "Partial CSV export",
    description: `The file holds the first ${exported.toLocaleString()} of ${totalLabel} results. ${nextStep}`,
  });
}

export function notifyCsvFailed(): void {
  toast({
    variant: "destructive",
    title: "CSV export failed",
    description: "The results couldn't be fetched. Try again in a moment.",
  });
}
