import { PageShell } from "@/components/PageShell";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from "@/components/ui/card";

export default function XmlSchema() {
  return (
    <PageShell
      size="xl"
      title="XML Schema"
      subtitle="Structure and element definitions for Pandects agreement XML outputs."
    >
      <div className="grid gap-6">
        <section
          aria-labelledby="xml-schema-overview"
          aria-describedby="xml-schema-overview-desc"
        >
          <Card className="border-border/60 bg-card">
            <CardHeader>
              <h2
                id="xml-schema-overview"
                className="text-xl font-semibold leading-none tracking-tight"
              >
                Overview
              </h2>
              <CardDescription id="xml-schema-overview-desc">
                The XML output is a single document tree per agreement. It
                contains metadata, optional page-type containers, and a
                structured body with articles and sections.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg border border-border/60 bg-muted/40 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Root Node
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground">
                    <span className="font-mono text-xs text-foreground">&lt;document&gt;</span>{" "}
                    with a required{" "}
                    <span className="font-mono text-xs text-foreground">uuid</span> attribute.
                  </p>
                </div>
                <div className="rounded-lg border border-border/60 bg-muted/40 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Page Containers
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Optional containers split by page type:{" "}
                    <span className="font-mono text-xs text-foreground">frontMatter</span>,{" "}
                    <span className="font-mono text-xs text-foreground">tableOfContents</span>,{" "}
                    <span className="font-mono text-xs text-foreground">body</span>,{" "}
                    <span className="font-mono text-xs text-foreground">sigPages</span>,{" "}
                    <span className="font-mono text-xs text-foreground">backMatter</span>.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section
          aria-labelledby="xml-schema-definition"
          aria-describedby="xml-schema-definition-desc"
        >
          <Card className="border-border/60 bg-card">
            <CardHeader>
              <h2
                id="xml-schema-definition"
                className="text-xl font-semibold leading-none tracking-tight"
              >
                Element Tree
              </h2>
              <CardDescription id="xml-schema-definition-desc">
                The tree below mirrors the current generator behavior.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                role="region"
                aria-label="XML element tree"
                tabIndex={0}
                className="rounded-lg border border-border/70 bg-muted/40 p-4 overflow-x-auto"
              >
                <pre className="whitespace-pre-wrap text-xs font-mono text-muted-foreground">
                  &lt;document uuid="agreement-uuid"&gt;
                  {"\n"}  &lt;metadata&gt;
                  {"\n"}    &lt;acquirer&gt;...&lt;/acquirer&gt;
                  {"\n"}    &lt;target&gt;...&lt;/target&gt;
                  {"\n"}    &lt;filingDate&gt;YYYY-MM-DD&lt;/filingDate&gt;
                  {"\n"}    &lt;url&gt;...&lt;/url&gt;
                  {"\n"}    &lt;sourceFormat&gt;html|txt&lt;/sourceFormat&gt;
                  {"\n"}  &lt;/metadata&gt;
                  {"\n"}  &lt;frontMatter&gt;...&lt;/frontMatter&gt;
                  {"\n"}  &lt;tableOfContents&gt;...&lt;/tableOfContents&gt;
                  {"\n"}  &lt;body&gt;
                  {"\n"}    &lt;article title="..." uuid="..." order="1" standardId="..."&gt;
                  {"\n"}      &lt;section title="..." uuid="..." order="1" standardId="..."&gt;
                  {"\n"}        &lt;text&gt;...&lt;/text&gt;
                  {"\n"}        &lt;pageUUID&gt;...&lt;/pageUUID&gt;
                  {"\n"}        &lt;definition term="..." standardID="..."&gt;
                  {"\n"}          &lt;text&gt;...&lt;/text&gt;
                  {"\n"}          &lt;pageUUID&gt;...&lt;/pageUUID&gt;
                  {"\n"}        &lt;/definition&gt;
                  {"\n"}      &lt;/section&gt;
                  {"\n"}    &lt;/article&gt;
                  {"\n"}  &lt;/body&gt;
                  {"\n"}  &lt;sigPages&gt;...&lt;/sigPages&gt;
                  {"\n"}  &lt;backMatter&gt;...&lt;/backMatter&gt;
                  {"\n"}&lt;/document&gt;
                </pre>
              </div>
            </CardContent>
          </Card>
        </section>

        <section
          aria-labelledby="xml-schema-elements"
          aria-describedby="xml-schema-elements-desc"
        >
          <Card className="border-border/60 bg-card">
            <CardHeader>
              <h2
                id="xml-schema-elements"
                className="text-xl font-semibold leading-none tracking-tight"
              >
                Element Notes
              </h2>
              <CardDescription id="xml-schema-elements-desc">
                Textual content is normalized into a small set of node types.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg border border-border/60 bg-muted/40 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Content Nodes
                  </div>
                  <p className="mt-2 text-sm text-foreground">
                    <span className="font-mono text-xs text-foreground">
                      text
                    </span>
                    ,{" "}
                    <span className="font-mono text-xs text-foreground">
                      definition
                    </span>{" "}
                    <span className="text-muted-foreground">(optional)</span>,{" "}
                    <span className="font-mono text-xs text-foreground">
                      pageUUID
                    </span>
                    ,{" "}
                    <span className="font-mono text-xs text-foreground">
                      page
                    </span>{" "}
                    <span className="text-muted-foreground">(optional)</span>.
                  </p>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Definitions are detected from quoted terms.{" "}
                    <span className="font-mono text-xs text-foreground">
                      pageUUID
                    </span>{" "}
                    is typically a sibling of{" "}
                    <span className="font-mono text-xs text-foreground">
                      text
                    </span>{" "}
                    and{" "}
                    <span className="font-mono text-xs text-foreground">
                      definition
                    </span>{" "}
                    nodes, and only moves inside{" "}
                    <span className="font-mono text-xs text-foreground">
                      definition
                    </span>{" "}
                    when a page break lands within a definition.{" "}
                    <span className="font-mono text-xs text-foreground">
                      page
                    </span>{" "}
                    appears only when a page number is explicitly tagged.
                  </p>
                </div>
                <div className="rounded-lg border border-border/60 bg-muted/40 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Structural Rules
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground">
                    <span className="font-mono text-xs text-foreground">
                      article
                    </span>{" "}
                    nodes live under{" "}
                    <span className="font-mono text-xs text-foreground">
                      body
                    </span>{" "}
                    and carry{" "}
                    <span className="font-mono text-xs text-foreground">
                      title
                    </span>
                    ,{" "}
                    <span className="font-mono text-xs text-foreground">
                      uuid
                    </span>
                    ,{" "}
                    <span className="font-mono text-xs text-foreground">
                      order
                    </span>
                    , and{" "}
                    <span className="font-mono text-xs text-foreground">
                      standardId
                    </span>{" "}
                    attributes.
                  </p>
                  <p className="mt-2 text-sm text-muted-foreground">
                    <span className="font-mono text-xs text-foreground">
                      section
                    </span>{" "}
                    nodes follow the same attributes and may appear directly
                    under{" "}
                    <span className="font-mono text-xs text-foreground">
                      body
                    </span>{" "}
                    if no article precedes them.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </PageShell>
  );
}
