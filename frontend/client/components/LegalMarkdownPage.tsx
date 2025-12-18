import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Remarkable } from "remarkable";
import { linkify } from "remarkable/linkify";
import { Link } from "react-router-dom";

import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";

type LegalMarkdownPageProps = {
  title: string;
  markdownPath: string;
  relatedLinks?: React.ReactNode;
};

type PreparedMarkdown = {
  markdown: string;
  subtitle?: string;
};

function sanitizeMarkdownHref(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return "#";
  if (trimmed.startsWith("#")) return trimmed;
  if (trimmed.startsWith("/")) return trimmed;
  if (trimmed.startsWith("//")) return "#";

  const hasScheme = /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed);
  if (hasScheme) {
    const scheme = trimmed.split(":", 1)[0]?.toLowerCase();
    if (scheme === "http" || scheme === "https" || scheme === "mailto" || scheme === "tel") {
      return trimmed;
    }
    return "#";
  }

  return trimmed;
}

function prepareMarkdownForPage(source: string): PreparedMarkdown {
  const withoutBom = source.replace(/^\uFEFF/, "");
  const lines = withoutBom.split(/\r?\n/);

  // Drop the leading H1 (we render the page title via PageShell).
  const firstNonEmptyIndex = lines.findIndex((line) => line.trim().length > 0);
  if (firstNonEmptyIndex !== -1 && lines[firstNonEmptyIndex]?.startsWith("# ")) {
    lines.splice(firstNonEmptyIndex, 1);
    if (lines[firstNonEmptyIndex]?.trim() === "") lines.splice(firstNonEmptyIndex, 1);
  }

  // Promote "Effective date: ..." into the subtitle.
  const effectiveDateIndex = lines.findIndex((line) =>
    /^Effective date:\s*/i.test(line.trim()),
  );
  let subtitle: string | undefined;
  if (effectiveDateIndex !== -1) {
    const raw = lines[effectiveDateIndex] ?? "";
    subtitle = raw
      .replace(/^Effective date:\s*/i, "Effective date: ")
      .replace(/\*\*/g, "")
      .trim();

    lines.splice(effectiveDateIndex, 1);
    if (lines[effectiveDateIndex]?.trim() === "") lines.splice(effectiveDateIndex, 1);
  }

  return { markdown: lines.join("\n").trim(), subtitle };
}

function slugifyHeading(text: string): string {
  const unescaped = text.replace(/\\([\\`*_{}\[\]()#+\-.!])/g, "$1");
  const withoutLinks = unescaped.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  const withoutFormatting = withoutLinks.replace(/[*_`~]/g, "");
  const withoutLeadingNumbering = withoutFormatting.replace(/^\s*\d+(\.\d+)*[.)]?\s*/, "");

  const slug = withoutLeadingNumbering
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return slug.length ? slug : "section";
}

function renderMarkdownToHtml(markdown: string): string {
  const env: { __usedHeadingIds?: Map<string, number> } = {};

  const md = new Remarkable({ html: false, typographer: true }).use(linkify);
  md.renderer.rules.link_open = (tokens: any[], idx: number, options: any, env: any, self: any) => {
    const token = tokens[idx];
    if (token && Array.isArray(token.attrs)) {
      const hrefIndex = token.attrs.findIndex(([name]: [string]) => name === "href");
      if (hrefIndex !== -1) {
        const hrefValue = token.attrs[hrefIndex]?.[1];
        if (typeof hrefValue === "string") {
          token.attrs[hrefIndex][1] = sanitizeMarkdownHref(hrefValue);
        }
      }
    }
    return self.renderToken(tokens, idx, options);
  };
  md.renderer.rules.heading_open = (tokens: any[], idx: number) => {
    const level = tokens[idx]?.hLevel ?? 2;
    const headingInline = tokens[idx + 1];
    const rawText = typeof headingInline?.content === "string" ? headingInline.content : "";

    const used = (env.__usedHeadingIds ??= new Map<string, number>());
    const base = slugifyHeading(rawText);
    const nextCount = (used.get(base) ?? 0) + 1;
    used.set(base, nextCount);
    const id = nextCount === 1 ? base : `${base}-${nextCount}`;

    return `<h${level} id="${id}">`;
  };

  return md.render(markdown, env);
}

async function fetchMarkdown(path: string): Promise<string> {
  const response = await fetch(path, { headers: { Accept: "text/markdown" } });
  if (!response.ok) throw new Error(`Failed to load ${path} (${response.status})`);
  return response.text();
}

export function LegalMarkdownPage({ title, markdownPath, relatedLinks }: LegalMarkdownPageProps) {
  const { data, error, isLoading } = useQuery({
    queryKey: ["legal-markdown", markdownPath],
    queryFn: () => fetchMarkdown(markdownPath),
    staleTime: Infinity,
    gcTime: Infinity,
  });

  const prepared = useMemo(() => (data ? prepareMarkdownForPage(data) : null), [data]);
  const html = useMemo(
    () => (prepared ? renderMarkdownToHtml(prepared.markdown) : ""),
    [prepared],
  );

  const defaultRelatedLinks =
    title === "Terms of Service" ? (
      <>
        <Link className="text-primary hover:underline" to="/privacy-policy">
          Privacy Policy
        </Link>{" "}
        and{" "}
        <Link className="text-primary hover:underline" to="/license">
          License
        </Link>
      </>
    ) : title === "Privacy Policy" ? (
      <>
        <Link className="text-primary hover:underline" to="/terms">
          Terms
        </Link>{" "}
        and{" "}
        <Link className="text-primary hover:underline" to="/license">
          License
        </Link>
      </>
    ) : null;

  const resolvedRelatedLinks = relatedLinks ?? defaultRelatedLinks;

  return (
    <PageShell size="md" title={title} subtitle={prepared?.subtitle}>
      <Card className="border-border/70 bg-background/70 p-8 backdrop-blur sm:p-10">
        <div className="mx-auto max-w-3xl">
          {isLoading && (
            <div className="text-sm text-muted-foreground" role="status" aria-live="polite">
              Loading…
            </div>
          )}
          {error instanceof Error && (
            <div className="text-sm text-destructive" role="alert">
              {error.message}
            </div>
          )}
          {!isLoading && !error && (
            <>
              <div
                className="prose prose-slate dark:prose-invert prose-headings:tracking-tight prose-h2:scroll-mt-24 prose-h2:text-xl prose-h2:font-semibold prose-h2:mt-10 prose-h2:mb-3 prose-h3:text-base prose-h3:font-semibold prose-h3:mt-6 prose-h3:mb-2 prose-p:leading-relaxed prose-a:text-primary prose-a:no-underline hover:prose-a:underline prose-a:underline-offset-4 prose-a:focus-visible:outline-none prose-a:focus-visible:ring-2 prose-a:focus-visible:ring-ring prose-a:focus-visible:ring-offset-2 prose-a:focus-visible:ring-offset-background prose-a:rounded-sm"
                dangerouslySetInnerHTML={{ __html: html }}
              />
              <p className="not-prose mt-6 text-sm text-muted-foreground">
                Download:{" "}
                <a
                  className="rounded-sm text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                  href={markdownPath}
                  aria-label={`Download ${title} as Markdown`}
                >
                  Markdown
                </a>
              </p>
              {resolvedRelatedLinks && (
                <p className="not-prose mt-4 text-sm text-muted-foreground">
                  Also see our {resolvedRelatedLinks}.
                </p>
              )}
            </>
          )}
        </div>
      </Card>
    </PageShell>
  );
}
