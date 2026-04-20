import { PageShell } from "@/components/PageShell";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from "@/components/ui/card";

const XML_ELEMENT_TREE = String.raw`<document uuid="agreement-uuid">
  <metadata>
    <agreementUuid>agreement-uuid</agreementUuid>
    <filingDate>YYYY-MM-DD</filingDate>
    <url>...</url>
    <sourceFormat>html|txt</sourceFormat>
  </metadata>
  <frontMatter>...</frontMatter>
  <tableOfContents>...</tableOfContents>
  <body>
    <article title="..." uuid="..." order="1" standardId="...">
      <section title="..." uuid="..." order="1" standardId="...">
        <text>...</text>
        <pageUUID>...</pageUUID>
        <definition term="..." standardID="...">
          <text>...</text>
          <pageUUID>...</pageUUID>
        </definition>
      </section>
    </article>
  </body>
  <sigPages>...</sigPages>
  <backMatter>...</backMatter>
</document>`;

export default function XmlSchema() {
  return (
    <PageShell
      size="xl"
      title="XML Schema"
    >
      <div className="grid gap-6">
        <section
          aria-labelledby="xml-schema-overview"
          aria-describedby="xml-schema-overview-desc"
        >
          <Card className="border-border bg-card">
            <CardHeader>
              <h2
                id="xml-schema-overview"
                className="text-xl font-semibold leading-none tracking-tight"
              >
                Overview
              </h2>
              <CardDescription id="xml-schema-overview-desc" className="text-base">
                The XML output is a single document tree per agreement. It
                contains metadata, optional page-type containers, and a
                structured body with articles and sections.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg border border-border bg-muted/40 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Root Node
                  </div>
                  <p className="mt-2 text-muted-foreground">
                    <span className="font-mono text-sm text-foreground">&lt;document&gt;</span>{" "}
                    with a required{" "}
                    <span className="font-mono text-sm text-foreground">uuid</span>{" "}
                    attribute. The same value is repeated in{" "}
                    <span className="font-mono text-sm text-foreground">metadata.agreementUuid</span>.
                  </p>
                </div>
                <div className="rounded-lg border border-border bg-muted/40 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Page Containers
                  </div>
                  <p className="mt-2 text-muted-foreground">
                    Optional containers split by page type:{" "}
                    <span className="font-mono text-sm text-foreground">frontMatter</span>,{" "}
                    <span className="font-mono text-sm text-foreground">tableOfContents</span>,{" "}
                    <span className="font-mono text-sm text-foreground">body</span>,{" "}
                    <span className="font-mono text-sm text-foreground">sigPages</span>,{" "}
                    <span className="font-mono text-sm text-foreground">backMatter</span>.
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
          <Card className="border-border bg-card">
            <CardHeader>
              <h2
                id="xml-schema-definition"
                className="text-xl font-semibold leading-none tracking-tight"
              >
                Element Tree
              </h2>
              <CardDescription id="xml-schema-definition-desc" className="text-base">
                The tree below mirrors the current generator behavior.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                role="region"
                aria-label="XML element tree"
                tabIndex={0}
                className="rounded-lg border border-border bg-muted/40 p-4"
              >
                <pre className="overflow-x-auto whitespace-pre-wrap break-words text-xs font-mono text-muted-foreground [overflow-wrap:anywhere] sm:text-sm sm:whitespace-pre">
                  {XML_ELEMENT_TREE}
                </pre>
              </div>
            </CardContent>
          </Card>
        </section>

        <section
          aria-labelledby="xml-schema-elements"
          aria-describedby="xml-schema-elements-desc"
        >
          <Card className="border-border bg-card">
            <CardHeader>
              <h2
                id="xml-schema-elements"
                className="text-xl font-semibold leading-none tracking-tight"
              >
                Element Notes
              </h2>
              <CardDescription id="xml-schema-elements-desc" className="text-base">
                Textual content is normalized into a small set of node types.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg border border-border bg-muted/40 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Metadata Nodes
                  </div>
                  <p className="mt-2 text-muted-foreground">
                    <span className="font-mono text-sm text-foreground">agreementUuid</span>,{" "}
                    <span className="font-mono text-sm text-foreground">filingDate</span>,{" "}
                    <span className="font-mono text-sm text-foreground">url</span>,{" "}
                    <span className="font-mono text-sm text-foreground">sourceFormat</span>.
                  </p>
                  <p className="mt-2 text-muted-foreground">
                    Party-name metadata is not embedded in the XML document.
                    Use agreement-level API fields when you need target or
                    acquirer names.
                  </p>
                </div>
                <div className="rounded-lg border border-border bg-muted/40 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Content Nodes
                  </div>
                  <p className="mt-2 text-muted-foreground">
                    <span className="font-mono text-sm text-foreground">text</span>,{" "}
                    <span className="font-mono text-sm text-foreground">
                      definition
                    </span>{" "}
                    <span className="text-muted-foreground">(optional)</span>,{" "}
                    <span className="font-mono text-sm text-foreground">
                      pageUUID
                    </span>
                    and{" "}
                    <span className="font-mono text-sm text-foreground">
                      page
                    </span>{" "}
                    <span className="text-muted-foreground">(optional)</span>.
                  </p>
                  <p className="mt-2 text-muted-foreground">
                    Definitions are detected from quoted terms.{" "}
                    <span className="font-mono text-sm text-foreground">
                      pageUUID
                    </span>{" "}
                    is typically a sibling of text and definition nodes, and
                    only moves inside definition when a page break lands within
                    a definition.{" "}
                    <span className="font-mono text-sm text-foreground">
                      page
                    </span>{" "}
                    appears only when a page number is explicitly tagged.
                  </p>
                </div>
                <div className="rounded-lg border border-border bg-muted/40 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Structural Rules
                  </div>
                  <p className="mt-2 text-muted-foreground">
                    <span className="font-mono text-sm text-foreground">
                      article
                    </span>{" "}
                    nodes live under{" "}
                    <span className="font-mono text-sm text-foreground">
                      body
                    </span>{" "}
                    and carry{" "}
                    <span className="font-mono text-sm text-foreground">
                      title
                    </span>
                    ,{" "}
                    <span className="font-mono text-sm text-foreground">
                      uuid
                    </span>
                    ,{" "}
                    <span className="font-mono text-sm text-foreground">
                      order
                    </span>
                    , and{" "}
                    <span className="font-mono text-sm text-foreground">
                      standardId
                    </span>{" "}
                    attributes.
                  </p>
                  <p className="mt-2 text-muted-foreground">
                    <span className="font-mono text-sm text-foreground">
                      section
                    </span>{" "}
                    nodes follow the same attributes and may appear directly
                    under{" "}
                    <span className="font-mono text-sm text-foreground">
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
