import React, { useEffect, useRef, useState } from "react";

import BrowserOnly from "@docusaurus/BrowserOnly";
import { useTypedSelector } from "@theme/ApiItem/hooks";

function colorForMethod(method: string) {
  switch (method.toLowerCase()) {
    case "get":
      return "primary";
    case "post":
      return "success";
    case "delete":
      return "danger";
    case "put":
      return "info";
    case "patch":
      return "warning";
    case "head":
      return "secondary";
    case "event":
      return "secondary";
    default:
      return undefined;
  }
}

export interface Props {
  method: string;
  path: string;
  context?: "endpoint" | "callback";
}

const LOCAL_API_URL = "http://localhost:5113";
const PROD_API_URL = "https://api.pandects.org";

function isLocalHostname(hostname: string): boolean {
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname.endsWith(".local")
  );
}

function resolveServerUrl(serverValue: any): string {
  if (serverValue?.url) {
    let resolvedUrl = serverValue.url.replace(/\/$/, "");
    if (serverValue.variables) {
      Object.keys(serverValue.variables).forEach((variable: string) => {
        resolvedUrl = resolvedUrl.replace(
          `{${variable}}`,
          serverValue.variables?.[variable]?.default ?? ""
        );
      });
    }
    return resolvedUrl;
  }
  return typeof window !== "undefined" && isLocalHostname(window.location.hostname)
    ? LOCAL_API_URL
    : PROD_API_URL;
}

function copyTextToClipboard(text: string) {
  if (!text) {
    return Promise.resolve();
  }

  if (navigator?.clipboard?.writeText) {
    return navigator.clipboard.writeText(text);
  }

  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.position = "fixed";
  textArea.style.left = "-9999px";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  document.execCommand("copy");
  document.body.removeChild(textArea);
  return Promise.resolve();
}

function renderUrlWithStyledParams(url: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  const paramRegex = /\{([^}]+)\}/g;
  let lastIndex = 0;
  let match;

  while ((match = paramRegex.exec(url)) !== null) {
    if (match.index > lastIndex) {
      parts.push(url.slice(lastIndex, match.index));
    }
    parts.push(
      <span key={match.index} className="openapi__url-param">
        {match[0]}
      </span>
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < url.length) {
    parts.push(url.slice(lastIndex));
  }

  return parts.length > 0 ? parts : url;
}

function MethodEndpoint({ method, path, context }: Props) {
  const serverValue = useTypedSelector((state: any) => state.server.value);
  const [copied, setCopied] = useState(false);
  const clearCopiedTimeoutRef = useRef<number | null>(null);

  const serverUrl = context === "callback" ? "" : resolveServerUrl(serverValue);
  const fullUrl = `${serverUrl}${path}`;

  useEffect(() => {
    return () => {
      if (clearCopiedTimeoutRef.current !== null) {
        window.clearTimeout(clearCopiedTimeoutRef.current);
      }
    };
  }, []);

  const handleCopyUrl = async () => {
    try {
      await copyTextToClipboard(fullUrl || path);
      setCopied(true);
      if (clearCopiedTimeoutRef.current !== null) {
        window.clearTimeout(clearCopiedTimeoutRef.current);
      }
      clearCopiedTimeoutRef.current = window.setTimeout(() => {
        setCopied(false);
      }, 1500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <>
      {method !== "event" && (
        <div className="openapi__route-heading">{path}</div>
      )}
      <div className="openapi__method-endpoint">
        <span className={"badge badge--" + colorForMethod(method)}>
          {method === "event" ? "Webhook" : method.toUpperCase()}
        </span>
        {method !== "event" && (
          <h2 className="openapi__method-endpoint-path">
            <BrowserOnly>{() => renderUrlWithStyledParams(fullUrl)}</BrowserOnly>
          </h2>
        )}
        {method !== "event" && (
          <button
            type="button"
            className="openapi__copy-url-btn"
            data-copied={copied ? "true" : "false"}
            aria-label="Copy endpoint URL"
            title={copied ? "Copied" : "Copy URL"}
            onClick={handleCopyUrl}
          >
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              aria-hidden="true"
              focusable="false"
            >
              <path
                fill="currentColor"
                d="M16 1H6a2 2 0 0 0-2 2v12h2V3h10V1zm3 4H10a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H10V7h9v14z"
              />
            </svg>
          </button>
        )}
      </div>
      <div className="openapi__divider" />
    </>
  );
}

export default MethodEndpoint;
