import { useState, useEffect } from "react";
import { Copy, Check } from "lucide-react";
import Navigation from "@/components/Navigation";
import { apiUrl } from "@/lib/api-config";

interface DumpInfo {
  manifest: string;
  sha256: string;
  sql: string;
  timestamp: string;
}

export default function BulkData() {
  const [dumps, setDumps] = useState<DumpInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copiedStates, setCopiedStates] = useState<{ [key: string]: boolean }>(
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
        console.error("Error fetching dumps:", err);
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

  const copyToClipboard = (text: string, id: string) => {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-9999px";
    textArea.style.top = "-9999px";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    try {
      document.execCommand("copy");
      setCopiedStates((prev) => ({ ...prev, [id]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [id]: false }));
      }, 2000);
    } catch (err) {
      // Silent fail
    } finally {
      document.body.removeChild(textArea);
    }
  };

  const downloadManifest = async (url: string) => {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const filename = url.split("/").pop()!;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (e) {
      console.error("Failed to download manifest", e);
    }
  };


  return (
    <div className="min-h-screen bg-cream">
      <Navigation />

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-material-text-primary mb-4">
            Bulk Data Downloads
          </h1>
          <p className="text-material-text-secondary max-w-3xl">
            Download complete database dumps of the Pandects
            dataset. All dumps are compressed MariaDB SQL files containing the
            full structured data.
          </p>
        </div>

        {/* Demo Code Blocks */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* API Call Example */}
          <div className="bg-white rounded-lg border border-material-divider p-6 min-w-0">
            <h3 className="text-lg font-semibold text-material-text-primary mb-3">
              Pull metadata for all dumps via API
            </h3>
            <div className="bg-gray-50 rounded p-4 text-xs font-mono relative group min-h-[85px] flex flex-col justify-center">
              <button
                onClick={() =>
                  copyToClipboard(
                    "curl https://pandects-api.fly.dev/api/dumps",
                    "api-call",
                  )
                }
                className="absolute top-2 right-2 p-1.5 rounded bg-white shadow-sm border border-gray-200 transition-opacity duration-200 hover:bg-gray-50 z-10"
                title="Copy to clipboard"
              >
                {copiedStates["api-call"] ? (
                  <Check className="w-3 h-3 text-green-600" />
                ) : (
                  <Copy className="w-3 h-3 text-gray-600" />
                )}
              </button>
              <div className="overflow-x-auto pb-2 flex-1 flex flex-col justify-center">
                <div>
                  <div className="text-gray-600 mb-2">
                    # API call to get dumps info
                  </div>
                  <div className="whitespace-nowrap pr-10">
                    curl https://pandects-api.fly.dev/api/dumps
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Download with wget */}
          <div className="bg-white rounded-lg border border-material-divider p-6 min-w-0">
            <h3 className="text-lg font-semibold text-material-text-primary mb-3">
              Download latest version with wget
            </h3>
            <div className="bg-gray-50 rounded p-4 text-xs font-mono relative group min-h-[85px] flex flex-col justify-center">
              <button
                onClick={() =>
                  latestSqlUrl &&
                  copyToClipboard(`wget ${latestSqlUrl}`, "wget-download")
                }
                className="absolute top-2 right-2 p-1.5 rounded bg-white shadow-sm border border-gray-200 hover:bg-gray-50 z-10"
                title="Copy to clipboard"
              >
                {copiedStates["wget-download"] ? (
                  <Check className="w-3 h-3 text-green-600" />
                ) : (
                  <Copy className="w-3 h-3 text-gray-600" />
                )}
              </button>
              <div className="overflow-x-auto pb-2 flex-1 flex flex-col justify-center">
                <div className="text-gray-600 mb-2"># Download latest dump</div>
                <div className="whitespace-nowrap pr-10">
                  {latestSqlUrl
                    ? `wget ${latestSqlUrl}`
                    : "Loading latest URL..."}
                </div>
              </div>
            </div>
          </div>

          {/* Verify the checksum */}
          <div className="bg-white rounded-lg border border-material-divider p-6 min-w-0">
            <h3 className="text-lg font-semibold text-material-text-primary mb-3">
              Verify the checksum
            </h3>
            <div className="bg-gray-50 rounded p-4 text-xs font-mono relative group min-h-[85px] flex flex-col justify-center">
              <button
                onClick={() =>
                  latestSha256 &&
                  copyToClipboard(
                    `echo "${latestSha256}  latest.sql.gz" | sha256sum -c -`,
                    "checksum-verify"
                  )
                }
                className="absolute top-2 right-2 p-1.5 rounded bg-white shadow-sm border border-gray-200 hover:bg-gray-50 z-10"
                title="Copy to clipboard"
              >
                {copiedStates["checksum-verify"] ? (
                  <Check className="w-3 h-3 text-green-600" />
                ) : (
                  <Copy className="w-3 h-3 text-gray-600" />
                )}
              </button>
              <div className="overflow-x-auto pb-2 flex-1 flex flex-col justify-center">
                <div className="text-gray-600 mb-2"># Verify file integrity</div>
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
        <div className="bg-material-blue-light border border-material-blue rounded-lg p-4 mb-8">
          <h3 className="text-lg font-semibold text-material-blue mb-2">
            About the SQL Dump
          </h3>
          <p className="text-material-text-secondary">
            The database dumps are in MariaDB SQL format. For installation,
            setup, and usage instructions, visit the{" "}
            <a
              href="https://mariadb.org/documentation/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-material-blue hover:underline"
            >
              MariaDB Documentation
            </a>
            .
          </p>
        </div>

        {/* Data Table */}
        <div className="bg-white rounded-lg border border-material-divider overflow-hidden">
          <div className="px-6 py-4 border-b border-material-divider">
            <h2 className="text-xl font-semibold text-material-text-primary">
              Available Downloads
            </h2>
            <p className="text-material-text-secondary mt-1">
              Database dumps hosted on Cloudflare, sorted by date (newest first)
            </p>
          </div>

          {loading ? (
            <div className="p-8 text-center">
              <div className="animate-spin inline-block w-6 h-6 border-2 border-material-blue border-t-transparent rounded-full"></div>
              <p className="text-material-text-secondary mt-2">
                Loading dumps...
              </p>
            </div>
          ) : error ? (
            <div className="p-8 text-center">
              <p className="text-red-600">{error}</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-material-surface">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-material-text-secondary uppercase tracking-wider">
                      Version
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-material-text-secondary uppercase tracking-wider">
                      SHA256 Hash
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-material-text-secondary uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-material-divider">
                  {dumps.map((dump, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex flex-col">
                          <span className="font-medium text-material-text-primary">
                            {formatTimestamp(dump.timestamp)}
                          </span>
                          {dump.timestamp === "latest" && (
                            <span className="text-xs text-green-600 font-medium">
                              Latest Version
                            </span>
                          )}
                          {dump.timestamp !== "latest" &&
                            latestSha256 &&
                            dump.sha256 === latestSha256 && (
                              <span className="text-xs text-green-600 font-medium">
                                Same as Latest
                              </span>
                            )}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center space-x-2">
                          <span className="font-mono text-sm text-material-text-secondary truncate max-w-xs">
                            {dump.sha256}
                          </span>
                          <button
                            onClick={() =>
                              copyToClipboard(dump.sha256, `sha-${index}`)
                            }
                            className="p-1 rounded hover:bg-gray-100 transition-colors"
                            title="Copy SHA256"
                          >
                            {copiedStates[`sha-${index}`] ? (
                              <Check className="w-3 h-3 text-green-600" />
                            ) : (
                              <Copy className="w-3 h-3 text-gray-600" />
                            )}
                          </button>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex space-x-2">
                          {/* direct SQL download (R2.dev supports cross‑origin downloads) */}
                          <a
                            href={dump.sql}
                            className="inline-flex items-center px-3 py-1 border border-material-blue text-material-blue hover:bg-material-blue hover:text-white rounded text-sm font-medium transition-colors"
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            Download SQL
                          </a>

                          {/* manifest: fetch+trigger download */}
                          <button
                            onClick={() => downloadManifest(dump.manifest)}
                            className="inline-flex items-center px-3 py-1 border border-gray-300 text-gray-700 hover:bg-gray-50 rounded text-sm font-medium transition-colors"
                          >
                            Manifest
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
