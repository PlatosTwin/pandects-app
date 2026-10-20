import type React from "react";
import { API_BASE_URL } from "@/lib/api-config";
import { getDocsUrl } from "@/lib/docs-url";
import { useDumps } from "@/hooks/use-insights";
import type { DumpInfo } from "@shared/insights";
import { trackEvent } from "@/lib/analytics";
import { logger } from "@/lib/logger";
import { PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ErrorState } from "@/components/StateMessage";
import { CodeBlock } from "@/components/ui/code-block";
import { CopyButton } from "@/components/ui/copy-button";
import { Eyebrow } from "@/components/ui/eyebrow";
import { ExternalLink } from "@/components/ui/external-link";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { LoadingSpinner } from "@/components/ui/loading-spinner";

const downloadManifest = async (url: string) => {
  try {
    const res = await fetch(url);
    if (!res.ok) {
      trackEvent("api_error", {
        endpoint: "bulk/manifest",
        status: res.status,
      });
      throw new Error(`HTTP ${res.status}`);
    }
    const blob = await res.blob();
    const filename = url.split("/").pop()!;
    const link = document.createElement("a");
    const objectUrl = URL.createObjectURL(blob);
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
  } catch (e) {
    if (import.meta.env.DEV) {
      logger.error("Failed to download manifest", e);
    }
    trackEvent("api_error", {
      endpoint: "bulk/manifest",
      kind:
        e instanceof TypeError && e.message.includes("fetch")
          ? "network"
          : "unknown",
    });
  }
};

function DumpActions({ dump, className }: { dump: DumpInfo; className: string }) {
  const trackDownload = (downloadType: "sql" | "manifest") => {
    trackEvent("bulk_download_click", {
      download_type: downloadType,
      dump_version: dump.timestamp,
      is_latest: dump.timestamp === "latest",
    });
    if (dump.timestamp !== "latest") {
      trackEvent("bulk_dated_download_click", {
        download_type: downloadType,
        dump_version: dump.timestamp,
      });
    }
  };

  return (
    <div className={className}>
      <Button asChild variant="outline" size="sm">
        <a
          href={dump.sql}
          onClick={() => trackDownload("sql")}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Download SQL (opens in a new tab)"
        >
          Download SQL
        </a>
      </Button>

      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => {
          trackDownload("manifest");
          void downloadManifest(dump.manifest);
        }}
      >
        Manifest
      </Button>
    </div>
  );
}

export default function BulkData() {
  const docsUrl = getDocsUrl();
  const {
    dumps,
    latestSha256,
    latestSqlUrl,
    isLoading: loading,
    error,
    apiError,
    refetch: refetchDumps,
  } = useDumps();

  const formatTimestamp = (timestamp: string): string => {
    if (timestamp === "latest") {
      return "Latest";
    }
    // Handle timestamp format like "2025-07-18_04-15"
    const parts = timestamp.split("_");
    if (parts.length === 2) {
      const datePart = parts[0];
      const timePart = parts[1].replace("-", ":");
      const dateTime = new Date(`${datePart}T${timePart}:00`);
      return dateTime.toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    }
    return timestamp;
  };

  const formatSize = (bytes?: number): string => {
    if (!bytes) return "—";
    const megabytes = bytes / (1024 * 1024);
    return `${megabytes.toFixed(1)} MB`;
  };

  const formatSha256 = (sha256: string) => {
    if (sha256.length <= 24) return sha256;
    return `${sha256.slice(0, 12)}…${sha256.slice(-12)}`;
  };

  const latestDump = dumps.find((dump) => dump.timestamp === "latest") ?? null;
  // The "latest" alias has no date of its own; its dated twin shares the hash.
  const latestDatedTwin = latestDump
    ? dumps.find(
        (dump) => dump.timestamp !== "latest" && dump.sha256 === latestDump.sha256,
      ) ?? null
    : null;
  const historyDumps = dumps.filter((dump) => dump.timestamp !== "latest");

  const apiCommand = `curl ${API_BASE_URL}/v1/dumps`;
  const wgetCommand = latestSqlUrl
    ? `wget ${latestSqlUrl}`
    : loading
      ? "Loading latest URL..."
      : "wget <latest SQL URL>";
  const checksumCommand = latestSha256
    ? `echo "${latestSha256}  latest.sql.gz" | sha256sum -c -`
    : loading
      ? "Loading latest SHA256..."
      : 'echo "<SHA256>  latest.sql.gz" | sha256sum -c -';
  const commandsUnavailable = !loading && (!latestSqlUrl || !latestSha256);

  const trackCopy = (copyTarget: string) => () => {
    trackEvent("bulk_copy_click", { copy_target: copyTarget });
  };

  return (
    <PageShell layout="data" title="Bulk Data">
      <div className="space-y-8">
        {/* Latest dump */}
        <section
          aria-labelledby="bulk-latest-heading"
          className="rounded-xl border border-border bg-card p-4 sm:p-6"
        >
          <h2
            id="bulk-latest-heading"
            className="text-lg font-semibold tracking-tight text-foreground"
          >
            Latest version
          </h2>
          {loading ? (
            <div className="mt-4 space-y-2" role="status" aria-live="polite">
              <span className="sr-only">Loading dumps...</span>
              <Skeleton className="h-5 w-64" />
              <Skeleton className="h-9 w-full max-w-xl" />
            </div>
          ) : error ? (
            <ErrorState
              className="mt-4"
              variant="inline"
              error={apiError ?? error}
              body="Could not load the list of downloads. Check your connection and try again."
              onRetry={() => void refetchDumps()}
            />
          ) : latestDump ? (
            <div className="mt-4 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <dl className="grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-[auto_auto_minmax(0,1fr)]">
                <div>
                  <Eyebrow asChild>
                    <dt>Version</dt>
                  </Eyebrow>
                  <dd className="mt-1 text-sm font-medium text-foreground">
                    {latestDatedTwin
                      ? formatTimestamp(latestDatedTwin.timestamp)
                      : formatTimestamp(latestDump.timestamp)}
                  </dd>
                </div>
                <div>
                  <Eyebrow asChild>
                    <dt>Download size</dt>
                  </Eyebrow>
                  <dd className="mt-1 text-sm text-foreground">
                    {formatSize(latestDump.size_bytes)}
                  </dd>
                </div>
                <div className="col-span-2 min-w-0 sm:col-span-1">
                  <Eyebrow asChild>
                    <dt>SHA256 hash</dt>
                  </Eyebrow>
                  <dd className="mt-0.5 flex items-center gap-1">
                    <span
                      className="truncate font-mono text-sm text-foreground/80"
                      title={latestDump.sha256}
                    >
                      {formatSha256(latestDump.sha256)}
                    </span>
                    <CopyButton value={latestDump.sha256} label="Copy SHA256" />
                  </dd>
                </div>
              </dl>
              <div className="flex flex-wrap items-center gap-2">
                <DumpActions dump={latestDump} className="flex flex-wrap gap-2" />
                <CopyButton
                  value={`wget ${latestDump.sql}`}
                  label="Copy wget command for latest SQL"
                  showLabel
                  idleText="Copy wget"
                  variant="outline"
                  className="h-9 text-foreground"
                  onCopied={trackCopy("wget_latest")}
                />
              </div>
            </div>
          ) : (
            <p className="mt-4 text-sm text-muted-foreground">No dumps found.</p>
          )}
        </section>

        {/* History */}
        <section
          aria-labelledby="bulk-history-heading"
          className="overflow-hidden rounded-xl border border-border bg-card"
        >
          <div className="border-b border-border px-4 py-4 sm:px-6">
            <h2
              id="bulk-history-heading"
              className="text-lg font-semibold tracking-tight text-foreground"
            >
              Available Downloads
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Database dumps hosted on Cloudflare, sorted by date (newest first)
            </p>
          </div>

          {loading ? (
            <div className="p-8 text-center" role="status" aria-live="polite">
              <LoadingSpinner size="lg" aria-label="Loading dumps" className="mx-auto" />
              <p className="mt-2 text-foreground/80">Loading dumps...</p>
            </div>
          ) : error ? (
            <ErrorState
              variant="inline"
              error={apiError ?? error}
              body="Could not load the list of downloads. Check your connection and try again."
              onRetry={() => void refetchDumps()}
            />
          ) : (
            <>
              <div className="hidden overflow-x-auto lg:block">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="px-6">Version</TableHead>
                      <TableHead className="px-6">SHA256 hash</TableHead>
                      <TableHead className="px-6">Download size</TableHead>
                      <TableHead className="px-6">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {historyDumps.map((dump) => (
                      <TableRow key={dump.timestamp}>
                        <TableCell className="whitespace-nowrap px-6 py-3">
                          <div className="flex flex-col">
                            <span className="font-medium text-foreground">
                              {formatTimestamp(dump.timestamp)}
                            </span>
                            {latestSha256 && dump.sha256 === latestSha256 && (
                              <span className="text-xs font-medium text-primary">
                                Same as latest
                              </span>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="px-6 py-3">
                          <div className="flex items-center gap-1">
                            <span
                              className="max-w-xs truncate font-mono text-sm text-foreground/80"
                              title={dump.sha256}
                            >
                              {formatSha256(dump.sha256)}
                            </span>
                            <CopyButton value={dump.sha256} label="Copy SHA256" />
                          </div>
                        </TableCell>
                        <TableCell className="whitespace-nowrap px-6 py-3 text-sm text-foreground/90">
                          {formatSize(dump.size_bytes)}
                        </TableCell>
                        <TableCell className="whitespace-nowrap px-6 py-3">
                          <DumpActions dump={dump} className="flex gap-2" />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              <ul className="divide-y divide-border lg:hidden">
                {historyDumps.map((dump) => (
                  <li key={dump.timestamp} className="space-y-3 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-sm font-semibold text-foreground">
                          {formatTimestamp(dump.timestamp)}
                        </div>
                        {latestSha256 && dump.sha256 === latestSha256 && (
                          <Badge variant="secondary" className="mt-1">
                            Same as latest
                          </Badge>
                        )}
                      </div>
                      <div className="text-xs text-foreground/90">
                        {formatSize(dump.size_bytes)}
                      </div>
                    </div>
                    <div>
                      <Eyebrow>SHA256 hash</Eyebrow>
                      <div className="mt-0.5 flex items-center gap-1">
                        <span className="truncate font-mono text-xs text-foreground/90">
                          {formatSha256(dump.sha256)}
                        </span>
                        <CopyButton value={dump.sha256} label="Copy SHA256" />
                      </div>
                    </div>
                    <DumpActions dump={dump} className="flex flex-wrap gap-2" />
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>

        {/* Usage commands */}
        <section aria-labelledby="bulk-usage-heading" className="space-y-3">
          <h2
            id="bulk-usage-heading"
            className="text-lg font-semibold tracking-tight text-foreground"
          >
            Usage
          </h2>
          <Tabs defaultValue="api">
            <TabsList className="h-auto max-w-full flex-wrap justify-start">
              <TabsTrigger value="api">API</TabsTrigger>
              <TabsTrigger value="wget">wget</TabsTrigger>
              <TabsTrigger value="checksum">Checksum</TabsTrigger>
            </TabsList>
            <TabsContent value="api" className="space-y-2">
              <p className="text-sm text-muted-foreground">
                Pull metadata for all dumps via API
              </p>
              <CodeBlock
                label="# API call to get dumps info"
                code={apiCommand}
                copyLabel="Copy API curl command"
                onCopied={trackCopy("api_call")}
              />
            </TabsContent>
            <TabsContent value="wget" className="space-y-2">
              <p className="text-sm text-muted-foreground">
                Download latest version with wget
              </p>
              <CodeBlock
                label="# Download latest dump"
                code={wgetCommand}
                copyLabel="Copy wget command for latest SQL"
                copyDisabled={!latestSqlUrl}
                onCopied={trackCopy("wget_latest")}
              />
            </TabsContent>
            <TabsContent value="checksum" className="space-y-2">
              <p className="text-sm text-muted-foreground">Verify the checksum</p>
              <CodeBlock
                label="# Verify file integrity"
                code={checksumCommand}
                copyLabel="Copy checksum verification command"
                copyDisabled={!latestSha256}
                onCopied={trackCopy("checksum_verify_latest")}
              />
            </TabsContent>
          </Tabs>
          {commandsUnavailable ? (
            <p className="text-sm text-muted-foreground">
              Pull metadata for all dumps via API:{" "}
              <ExternalLink
                href={`${API_BASE_URL}/v1/dumps`}
                className="font-mono text-primary underline underline-offset-2"
              >
                {`${API_BASE_URL}/v1/dumps`}
              </ExternalLink>
            </p>
          ) : null}
        </section>

        {/* Database Info */}
        <section
          aria-labelledby="bulk-about-heading"
          className="rounded-xl border border-border bg-card p-4 sm:p-6"
        >
          <h2
            id="bulk-about-heading"
            className="mb-2 text-lg font-semibold tracking-tight text-foreground"
          >
            About the SQL Dump
          </h2>
          <p className="prose prose-copy max-w-none">
            The database dumps are in MariaDB SQL format. For installation, setup,
            and usage instructions, visit the{" "}
            <ExternalLink href="https://mariadb.org/documentation/">
              MariaDB Documentation
            </ExternalLink>
            . For a table-by-table reference of what the dump contains, see the{" "}
            <ExternalLink href={`${docsUrl}/docs/guides/bulk-data-schema`}>
              Bulk Data Schema guide
            </ExternalLink>
            ; column-level documentation and an interactive ER diagram are on{" "}
            <ExternalLink href="https://dbdocs.io/nmbogdan/Pandects">dbdocs.io</ExternalLink>
            . Schema and data changes between releases are listed in the{" "}
            <ExternalLink href={`${docsUrl}/docs/guides/changelog`}>Changelog</ExternalLink>
            , which records the SHA256 of the dump each change shipped in.
          </p>
        </section>
      </div>
    </PageShell>
  );
}

/** Pulls the HTTP status out of "Failed to fetch dumps: 403" so ErrorState can map it. */
