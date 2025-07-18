import { useState, useEffect } from "react";
import Navigation from "@/components/Navigation";

interface DumpInfo {
  sha256: string;
  size: number;
  date: string;
  url: string;
}

export default function BulkData() {
  const [dumps, setDumps] = useState<DumpInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // TODO: Replace with actual API call when dumps endpoint is available
    // For now, showing placeholder data
    const mockData: DumpInfo[] = [
      {
        sha256:
          "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
        size: 1073741824, // 1GB in bytes
        date: "2024-01-15",
        url: "https://dumps.example.com/pandects-2024-01-15.sql.gz",
      },
      {
        sha256:
          "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567",
        size: 1048576000, // ~1GB in bytes
        date: "2024-01-01",
        url: "https://dumps.example.com/pandects-2024-01-01.sql.gz",
      },
    ];

    // Simulate API call
    setTimeout(() => {
      setDumps(mockData);
      setLoading(false);
    }, 500);
  }, []);

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatDate = (dateStr: string): string => {
    return new Date(dateStr).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
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
            Download complete database dumps of the Pandects merger agreement
            dataset. All dumps are compressed MariaDB SQL files containing the
            full structured data.
          </p>
        </div>

        {/* Demo Code Blocks */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* API Call Example */}
          <div className="bg-white rounded-lg border border-material-divider p-6">
            <h3 className="text-lg font-semibold text-material-text-primary mb-3">
              Get Latest Version
            </h3>
            <div className="bg-gray-50 rounded p-3 text-xs font-mono overflow-x-auto">
              <div className="text-gray-600 mb-2">
                # API call to get latest dump info
              </div>
              <div className="whitespace-pre-line">
                curl https://api.pandects.app/\{"\n"} api/dumps/latest
              </div>
            </div>
          </div>

          {/* wget Download Example */}
          <div className="bg-white rounded-lg border border-material-divider p-6">
            <h3 className="text-lg font-semibold text-material-text-primary mb-3">
              Download with wget
            </h3>
            <div className="bg-gray-50 rounded p-3 text-xs font-mono overflow-x-auto">
              <div className="text-gray-600 mb-2"># Download latest dump</div>
              <div className="whitespace-pre-line">
                wget https://dumps.pandects.app/\{"\n"} latest.sql.gz
              </div>
            </div>
          </div>

          {/* Checksum Verification */}
          <div className="bg-white rounded-lg border border-material-divider p-6">
            <h3 className="text-lg font-semibold text-material-text-primary mb-3">
              Verify Checksum
            </h3>
            <div className="bg-gray-50 rounded p-3 text-xs font-mono overflow-x-auto">
              <div className="text-gray-600 mb-2"># Verify file integrity</div>
              <div className="whitespace-pre-line">
                echo "sha256_hash filename.sql.gz" |\{"\n"} sha256sum -c
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
                      Date
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-material-text-secondary uppercase tracking-wider">
                      SHA256 Hash
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-material-text-secondary uppercase tracking-wider">
                      Size
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-material-divider">
                  {dumps.map((dump, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <a
                          href={dump.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-material-blue hover:underline font-medium"
                        >
                          {formatDate(dump.date)}
                        </a>
                      </td>
                      <td className="px-6 py-4">
                        <span className="font-mono text-sm text-material-text-secondary">
                          {dump.sha256}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-material-text-primary">
                          {formatBytes(dump.size)}
                        </span>
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
