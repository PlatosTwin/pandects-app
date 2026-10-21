/**
 * Dump versions look like "public_2026-09-09_16-08-00": the push script's
 * local wall-clock time with no zone. They are read as UTC so the viewer's
 * offset can't move a dump a day.
 */
const DUMP_TIMESTAMP = /(\d{4})-(\d{2})-(\d{2})(?:_(\d{2})-(\d{2})(?:-(\d{2}))?)?$/;

const DUMP_DATE_FORMATTER = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "long",
  day: "numeric",
  timeZone: "UTC",
});

/** Milliseconds for a dated dump version, or null ("latest", unknown formats). */
export function parseDumpTimestamp(timestamp: string): number | null {
  const match = DUMP_TIMESTAMP.exec(timestamp);
  if (!match) return null;
  const [, year, month, day, hour = "0", minute = "0", second = "0"] = match;
  return Date.UTC(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second),
  );
}

/** "September 9, 2026" for a dated version; "Latest"; else the raw value. */
export function formatDumpTimestamp(timestamp: string): string {
  if (timestamp === "latest") return "Latest";
  const parsed = parseDumpTimestamp(timestamp);
  return parsed === null ? timestamp : DUMP_DATE_FORMATTER.format(new Date(parsed));
}
