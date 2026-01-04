import { useState, useEffect } from "react";
import { Copy, Check } from "lucide-react";
import { API_BASE_URL, apiUrl } from "@/lib/api-config";
import { trackEvent } from "@/lib/analytics";
import { PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

async function showToast(kind: "success" | "error", message: string) {
  if (typeof window === "undefined") return;
  const { toast } = await import("sonner");
  if (kind === "success") toast.success(message);
  else toast.error(message);
}

interface DumpInfo {
  manifest: string;
  sha256: string;
  sha256_url?: string;
  sql: string;
  timestamp: string;
  size_bytes?: number;
  warning?: string;
}

export default function BulkData() {
  const [dumps, setDumps] = useState<DumpInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>(
    {},
  );
  const [latestSha256, setLatestSha256] = useState<string | null>(null);
  const [latestSqlUrl, setLatestSqlUrl] = useState<string | null>(null);

  useEffect(() => {
    const fetchDumps = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(apiUrl("api/dumps"));
        if (!response.ok) {
          trackEvent("api_error", {
            endpoint: "api/dumps",
            status: response.status,
            status_text: response.statusText,
          });
          throw new Error(`Failed to fetch dumps: ${response.status}`);
        }

        const data: DumpInfo[] = await response.json();

        // Find the latest version's SHA256
        const latest = data.find((dump) => dump.timestamp === "latest");
        const latestHash = latest?.sha256 || null;
        setLatestSha256(latestHash);
        setLatestSqlUrl(latest?.sql || null);

        // Sort so 'latest' is always first, then sort others by timestamp
        const sortedData = data.sort((a, b) => {
          if (a.timestamp === "latest") return -1;
          if (b.timestamp === "latest") return 1;
          // For non-latest items, sort by timestamp descending (newest first)
          return (
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
          );
        });

        setDumps(sortedData);
      } catch (err) {
        if (import.meta.env.DEV) {
          console.error("Error fetching dumps:", err);
        }
        trackEvent("api_error", {
          endpoint: "api/dumps",
          kind:
            err instanceof TypeError && err.message.includes("fetch")
              ? "network"
              : "unknown",
        });
        setError(err instanceof Error ? err.message : "Failed to load dumps");
      } finally {
        setLoading(false);
      }
    };

    fetchDumps();
  }, []);

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

  const copyToClipboard = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedStates((prev) => ({ ...prev, [id]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [id]: false }));
      }, 2000);
      void showToast("success", "Copied to clipboard");
    } catch (err) {
      void showToast("error", "Copy failed");
    }
  };

  const formatSha256 = (sha256: string) => {
    if (sha256.length <= 24) return sha256;
    return `${sha256.slice(0, 12)}…${sha256.slice(-12)}`;
  };

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
        console.error("Failed to download manifest", e);
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

  return (
      <PageShell
      size="xl"
      title="Bulk Data Downloads"
      subtitle={
        <span className="prose max-w-none">
          Download complete database dumps of the Pandects dataset. All dumps
          are compressed MariaDB SQL files containing the full structured data.
          For database documentation, see{" "}
          <a
            href="https://dbdocs.io/nmbogdan/Pandects"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="dbdocs (opens in a new tab)"
            className="underline underline-offset-2"
          >
            dbdocs
          </a>
          .
        </span>
      }
    >
      {/* Demo Code Blocks */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* API Call Example */}
        <div className="bg-card rounded-lg border border-border p-6 min-w-0">
          <h2 className="text-lg font-semibold text-foreground mb-3">
            Pull metadata for all dumps via API
          </h2>
          <div className="bg-muted/40 rounded p-4 text-xs font-mono relative group min-h-[85px] flex flex-col justify-center">
            <button
              type="button"
              onClick={() => {
                trackEvent("bulk_copy_click", { copy_target: "api_call" });
                void copyToClipboard(
                  `curl ${API_BASE_URL}/api/dumps`,
                  "api-call",
                );
              }}
              className="absolute top-2 right-2 p-1.5 rounded bg-background shadow-sm border border-border transition-opacity duration-200 hover:bg-accent z-10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              title="Copy to clipboard"
              aria-label="Copy API curl command"
            >
              {copiedStates["api-call"] ? (
                <Check className="w-3 h-3 text-primary" aria-hidden="true" />
              ) : (
                <Copy className="w-3 h-3 text-muted-foreground" aria-hidden="true" />
              )}
            </button>
            <div className="overflow-x-auto pb-2 flex-1 flex flex-col justify-center">
              <div>
                <div className="text-muted-foreground mb-2">
                  # API call to get dumps info
                </div>
                <div className="whitespace-nowrap pr-10">
                  curl {API_BASE_URL}/api/dumps
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Download with wget */}
        <div className="bg-card rounded-lg border border-border p-6 min-w-0">
          <h2 className="text-lg font-semibold text-foreground mb-3">
            Download latest version with wget
          </h2>
          <div className="bg-muted/40 rounded p-4 text-xs font-mono relative group min-h-[85px] flex flex-col justify-center">
            <button
              type="button"
              onClick={() => {
                if (!latestSqlUrl) return;
                trackEvent("bulk_copy_click", { copy_target: "wget_latest" });
                void copyToClipboard(`wget ${latestSqlUrl}`, "wget-download");
              }}
              className="absolute top-2 right-2 p-1.5 rounded bg-background shadow-sm border border-border hover:bg-accent z-10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              title="Copy to clipboard"
              aria-label="Copy wget command for latest SQL"
            >
              {copiedStates["wget-download"] ? (
                <Check className="w-3 h-3 text-primary" aria-hidden="true" />
              ) : (
                <Copy className="w-3 h-3 text-muted-foreground" aria-hidden="true" />
              )}
            </button>
            <div className="overflow-x-auto pb-2 flex-1 flex flex-col justify-center">
              <div className="text-muted-foreground mb-2">
                # Download latest dump
              </div>
              <div className="whitespace-nowrap pr-10">
                {latestSqlUrl
                  ? `wget ${latestSqlUrl}`
                  : "Loading latest URL..."}
              </div>
            </div>
          </div>
        </div>

        {/* Verify the checksum */}
        <div className="bg-card rounded-lg border border-border p-6 min-w-0">
          <h2 className="text-lg font-semibold text-foreground mb-3">
            Verify the checksum
          </h2>
          <div className="bg-muted/40 rounded p-4 text-xs font-mono relative group min-h-[85px] flex flex-col justify-center">
            <button
              type="button"
              onClick={() => {
                if (!latestSha256) return;
                trackEvent("bulk_copy_click", {
                  copy_target: "checksum_verify_latest",
                });
                void copyToClipboard(
                  `echo "${latestSha256}  latest.sql.gz" | sha256sum -c -`,
                  "checksum-verify",
                );
              }}
              className="absolute top-2 right-2 p-1.5 rounded bg-background shadow-sm border border-border hover:bg-accent z-10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              title="Copy to clipboard"
              aria-label="Copy checksum verification command"
            >
              {copiedStates["checksum-verify"] ? (
                <Check className="w-3 h-3 text-primary" aria-hidden="true" />
              ) : (
                <Copy className="w-3 h-3 text-muted-foreground" aria-hidden="true" />
              )}
            </button>
            <div className="overflow-x-auto pb-2 flex-1 flex flex-col justify-center">
              <div className="text-muted-foreground mb-2">
                # Verify file integrity
              </div>
              <div className="whitespace-nowrap pr-10">
                {latestSha256
                  ? `echo "${latestSha256}  latest.sql.gz" | sha256sum -c -`
                  : "Loading latest SHA256..."}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Database Info */}
      <div className="bg-primary/10 border border-primary/20 rounded-lg p-4 mb-8">
        <h2 className="text-lg font-semibold text-foreground mb-2">
          About the SQL Dump
        </h2>
        <p className="text-foreground prose max-w-none">
          The database dumps are in MariaDB SQL format. For installation, setup,
          and usage instructions, visit the{" "}
          <a
            href="https://mariadb.org/documentation/"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="MariaDB Documentation (opens in a new tab)"
            className="underline underline-offset-2"
          >
            MariaDB Documentation
          </a>
          .
        </p>
      </div>

      {/* Data Table */}
      <div className="min-h-[420px] overflow-hidden rounded-lg border border-border bg-card lg:min-h-[520px]">
        <div className="px-6 py-4 border-b border-border">
          <h2 className="text-xl font-semibold text-foreground">
            Available Downloads
          </h2>
          <p className="text-muted-foreground mt-1">
            Database dumps hosted on Cloudflare, sorted by date (newest first)
          </p>
        </div>

        {loading ? (
          <div className="p-8 text-center" role="status" aria-live="polite">
            <div className="animate-spin inline-block w-6 h-6 border-2 border-primary border-t-transparent rounded-full"></div>
            <p className="text-muted-foreground mt-2">Loading dumps...</p>
          </div>
        ) : error ? (
          <div className="p-8 text-center" role="alert">
            <p className="text-destructive">{error}</p>
          </div>
        ) : (
          <>
            <div className="hidden lg:block overflow-x-auto">
              <Table>
                <TableHeader className="bg-muted">
                  <TableRow>
                  <TableHead className="px-6 py-3 text-left text-xs font-medium text-foreground uppercase tracking-wider">
                    Version
                  </TableHead>
                  <TableHead className="px-6 py-3 text-left text-xs font-medium text-foreground uppercase tracking-wider">
                    SHA256 hash
                  </TableHead>
                  <TableHead className="px-6 py-3 text-left text-xs font-medium text-foreground uppercase tracking-wider">
                    Download size
                  </TableHead>
                  <TableHead className="px-6 py-3 text-left text-xs font-medium text-foreground uppercase tracking-wider">
                    Actions
                  </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody className="divide-y divide-border">
                  {dumps.map((dump, index) => (
                    <TableRow key={index} className="hover:bg-muted/40">
                      <TableCell className="px-6 py-4 whitespace-nowrap">
                        <div className="flex flex-col">
                          <span className="font-medium text-foreground">
                            {formatTimestamp(dump.timestamp)}
                          </span>
                          {dump.timestamp === "latest" && (
                            <span className="text-xs text-primary font-medium">
                              Latest version
                            </span>
                          )}
                          {dump.timestamp !== "latest" &&
                            latestSha256 &&
                            dump.sha256 === latestSha256 && (
                              <span className="text-xs text-primary font-medium">
                                Same as latest
                              </span>
                            )}
                        </div>
                      </TableCell>
                      <TableCell className="px-6 py-4">
                        <div className="flex items-center space-x-2">
                          <span className="font-mono text-sm text-muted-foreground truncate max-w-xs">
                            <span title={dump.sha256} aria-label={dump.sha256}>
                              {formatSha256(dump.sha256)}
                            </span>
                          </span>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            onClick={() =>
                              copyToClipboard(dump.sha256, `sha-${index}`)
                            }
                            className="h-7 w-7"
                            title="Copy SHA256"
                            aria-label="Copy SHA256"
                          >
                            {copiedStates[`sha-${index}`] ? (
                              <Check className="w-3 h-3 text-primary" aria-hidden="true" />
                            ) : (
                              <Copy className="w-3 h-3 text-muted-foreground" aria-hidden="true" />
                            )}
                          </Button>
                        </div>
                      </TableCell>
                      <TableCell className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm text-muted-foreground">
                          {formatSize(dump.size_bytes)}
                        </span>
                      </TableCell>
                      <TableCell className="px-6 py-4 whitespace-nowrap">
                        <div className="flex space-x-2">
                          <Button asChild variant="outline" size="sm">
                            <a
                              href={dump.sql}
                              onClick={() => {
                                trackEvent("bulk_download_click", {
                                  download_type: "sql",
                                  dump_version: dump.timestamp,
                                  is_latest: dump.timestamp === "latest",
                                });
                                if (dump.timestamp !== "latest") {
                                  trackEvent("bulk_dated_download_click", {
                                    download_type: "sql",
                                    dump_version: dump.timestamp,
                                  });
                                }
                              }}
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
                              trackEvent("bulk_download_click", {
                                download_type: "manifest",
                                dump_version: dump.timestamp,
                                is_latest: dump.timestamp === "latest",
                              });
                              if (dump.timestamp !== "latest") {
                                trackEvent("bulk_dated_download_click", {
                                  download_type: "manifest",
                                  dump_version: dump.timestamp,
                                });
                              }
                              void downloadManifest(dump.manifest);
                            }}
                          >
                            Manifest
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            <div className="lg:hidden p-4 space-y-4">
              {dumps.map((dump, index) => (
                <Card key={`mobile-${index}`} className="border-border/60">
                  <CardContent className="space-y-3 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-sm font-semibold text-foreground">
                          {formatTimestamp(dump.timestamp)}
                        </div>
                        {dump.timestamp === "latest" && (
                          <Badge variant="secondary" className="mt-1">
                            Latest version
                          </Badge>
                        )}
                        {dump.timestamp !== "latest" &&
                          latestSha256 &&
                          dump.sha256 === latestSha256 && (
                            <Badge variant="secondary" className="mt-1">
                              Same as latest
                            </Badge>
                          )}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {formatSize(dump.size_bytes)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        SHA256 hash
                      </div>
                      <div className="mt-1 flex items-center gap-2">
                        <span className="font-mono text-xs text-muted-foreground">
                          {formatSha256(dump.sha256)}
                        </span>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          onClick={() =>
                            copyToClipboard(dump.sha256, `sha-${index}`)
                          }
                          className="h-7 w-7"
                          title="Copy SHA256"
                          aria-label="Copy SHA256"
                        >
                          {copiedStates[`sha-${index}`] ? (
                            <Check className="w-3 h-3 text-primary" aria-hidden="true" />
                          ) : (
                            <Copy className="w-3 h-3 text-muted-foreground" aria-hidden="true" />
                          )}
                        </Button>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <Button asChild variant="outline" size="sm">
                        <a
                          href={dump.sql}
                          onClick={() => {
                            trackEvent("bulk_download_click", {
                              download_type: "sql",
                              dump_version: dump.timestamp,
                              is_latest: dump.timestamp === "latest",
                            });
                            if (dump.timestamp !== "latest") {
                              trackEvent("bulk_dated_download_click", {
                                download_type: "sql",
                                dump_version: dump.timestamp,
                              });
                            }
                          }}
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
                          trackEvent("bulk_download_click", {
                            download_type: "manifest",
                            dump_version: dump.timestamp,
                            is_latest: dump.timestamp === "latest",
                          });
                          if (dump.timestamp !== "latest") {
                            trackEvent("bulk_dated_download_click", {
                              download_type: "manifest",
                              dump_version: dump.timestamp,
                            });
                          }
                          void downloadManifest(dump.manifest);
                        }}
                      >
                        Manifest
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </>
        )}
      </div>
    </PageShell>
  );
}
