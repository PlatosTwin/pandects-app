import { Link } from "react-router-dom";

import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { prepareLegalMarkdownForPage, renderLegalMarkdownToHtml } from "@/lib/legal-markdown";

type LegalMarkdownPageProps = {
  title: string;
  markdown: string;
  downloadHref?: string | null;
  transformHtml?: ((html: string) => string) | null;
  relatedLinks?: React.ReactNode;
};

export function LegalMarkdownPage({
  title,
  markdown,
  downloadHref,
  transformHtml,
  relatedLinks,
}: LegalMarkdownPageProps) {
  const prepared = prepareLegalMarkdownForPage(markdown);
  const renderedHtml = renderLegalMarkdownToHtml(prepared.markdown);
  const html = transformHtml ? transformHtml(renderedHtml) : renderedHtml;

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
        {", and our "}
        <Link className="text-primary hover:underline" to="/license/software">
          software (GPLv3)
        </Link>{" "}
        and{" "}
        <Link className="text-primary hover:underline" to="/license/data">
          data (ODbL)
        </Link>{" "}
        licenses
      </>
    ) : title === "Privacy Policy" ? (
      <>
        <Link className="text-primary hover:underline" to="/terms">
          Terms of Service
        </Link>{" "}
        and{" "}
        <Link className="text-primary hover:underline" to="/license">
          License
        </Link>
        {", and our "}
        <Link className="text-primary hover:underline" to="/license/software">
          software (GPLv3)
        </Link>{" "}
        and{" "}
        <Link className="text-primary hover:underline" to="/license/data">
          data (ODbL)
        </Link>{" "}
        licenses
      </>
    ) : title === "License" ? (
      <>
        <Link className="text-primary hover:underline" to="/terms">
          Terms of Service
        </Link>{" "}
        and{" "}
        <Link className="text-primary hover:underline" to="/privacy-policy">
          Privacy Policy
        </Link>
        {", and our "}
        <Link className="text-primary hover:underline" to="/license/software">
          software (GPLv3)
        </Link>{" "}
        and{" "}
        <Link className="text-primary hover:underline" to="/license/data">
          data (ODbL)
        </Link>{" "}
        licenses
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
          {downloadHref && (
            <p className="not-prose mt-6 text-sm text-muted-foreground">
              Download:{" "}
              <a
                className="rounded-sm text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
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
      </Card>
    </PageShell>
  );
}
