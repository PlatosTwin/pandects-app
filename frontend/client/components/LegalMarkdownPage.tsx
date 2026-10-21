import { Link } from "react-router-dom";

import { PageShell } from "@/components/PageShell";
import { prepareLegalMarkdownForPage, renderLegalMarkdownToHtml } from "@/lib/legal-markdown";

// The two sibling core-legal pages each page cross-links, ahead of the
// shared software/data license links.
const RELATED_PAGE_LINKS = {
  terms: [
    { to: "/privacy-policy", label: "Privacy Policy" },
    { to: "/license", label: "License" },
  ],
  privacy: [
    { to: "/terms", label: "Terms of Service" },
    { to: "/license", label: "License" },
  ],
  license: [
    { to: "/terms", label: "Terms of Service" },
    { to: "/privacy-policy", label: "Privacy Policy" },
  ],
} as const;

type LegalPageKey = keyof typeof RELATED_PAGE_LINKS;

type LegalMarkdownPageProps = {
  title: string;
  markdown: string;
  pageKey?: LegalPageKey;
  downloadHref?: string | null;
  transformHtml?: ((html: string) => string) | null;
  relatedLinks?: React.ReactNode;
};

export function LegalMarkdownPage({
  title,
  markdown,
  pageKey,
  downloadHref,
  transformHtml,
  relatedLinks,
}: LegalMarkdownPageProps) {
  const prepared = prepareLegalMarkdownForPage(markdown);
  const renderedHtml = renderLegalMarkdownToHtml(prepared.markdown);
  // The page title is the only h1; headings the markdown marks as h1 render as h2.
  // Code blocks are focusable so keyboard users can scroll them.
  const demotedHtml = renderedHtml
    .replace(/<(\/?)h1(?=[\s>])/gi, "<$1h2")
    .replace(/<pre(?=[\s>])/gi, '<pre tabindex="0"');
  const html = transformHtml ? transformHtml(demotedHtml) : demotedHtml;

  const defaultRelatedLinks = pageKey ? (
    <>
      <Link
        className="text-primary underline underline-offset-4"
        to={RELATED_PAGE_LINKS[pageKey][0].to}
      >
        {RELATED_PAGE_LINKS[pageKey][0].label}
      </Link>{" "}
      and{" "}
      <Link
        className="text-primary underline underline-offset-4"
        to={RELATED_PAGE_LINKS[pageKey][1].to}
      >
        {RELATED_PAGE_LINKS[pageKey][1].label}
      </Link>
      {", and our "}
      <Link className="text-primary underline underline-offset-4" to="/license/software">
        software (GPLv3)
      </Link>{" "}
      and{" "}
      <Link className="text-primary underline underline-offset-4" to="/license/data">
        data (ODbL)
      </Link>{" "}
      licenses
    </>
  ) : null;

  const resolvedRelatedLinks = relatedLinks ?? defaultRelatedLinks;

  return (
    <PageShell layout="docs" title={title} description={prepared?.subtitle}>
      <div>
          <div
            className="prose prose-slate max-w-none dark:prose-invert prose-headings:tracking-tight prose-h2:scroll-mt-24 prose-h2:text-xl prose-h2:font-semibold prose-h2:mt-10 prose-h2:mb-3 prose-h3:text-base prose-h3:font-semibold prose-h3:mt-6 prose-h3:mb-2 prose-p:leading-relaxed [&_pre:focus-visible]:outline-none [&_pre:focus-visible]:ring-2 [&_pre:focus-visible]:ring-ring"
            dangerouslySetInnerHTML={{ __html: html }}
          />
          {downloadHref && (
            <p className="not-prose mt-6 text-sm text-muted-foreground">
              Download:{" "}
              <a
                className="rounded-sm text-primary underline underline-offset-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                href={downloadHref}
                aria-label={`Download ${title} as Markdown`}
              >
                Markdown
              </a>
            </p>
          )}
          {resolvedRelatedLinks && (
            <p className="not-prose mt-4 text-sm text-muted-foreground">
              Also see our {resolvedRelatedLinks}.
            </p>
          )}
      </div>
    </PageShell>
  );
}
