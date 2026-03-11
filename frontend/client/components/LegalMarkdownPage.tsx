import { Link } from "react-router-dom";

import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { prepareLegalMarkdownForPage, renderLegalMarkdownToHtml } from "@/lib/legal-markdown";

type LegalMarkdownPageProps = {
  title: string;
  markdownPath: string;
  markdown: string;
  relatedLinks?: React.ReactNode;
};

export function LegalMarkdownPage({
  title,
  markdownPath,
  markdown,
  relatedLinks,
}: LegalMarkdownPageProps) {
  const prepared = prepareLegalMarkdownForPage(markdown);
  const html = renderLegalMarkdownToHtml(prepared.markdown);

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
      <Card className="border-border/60 bg-background/70 p-8 backdrop-blur sm:p-10">
        <div className="mx-auto max-w-3xl">
          <div
            className="prose prose-slate dark:prose-invert prose-headings:tracking-tight prose-h2:scroll-mt-24 prose-h2:text-xl prose-h2:font-semibold prose-h2:mt-10 prose-h2:mb-3 prose-h3:text-base prose-h3:font-semibold prose-h3:mt-6 prose-h3:mb-2 prose-p:leading-relaxed"
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
        </div>
      </Card>
    </PageShell>
  );
}
