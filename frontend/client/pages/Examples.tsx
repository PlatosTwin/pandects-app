import { PageShell } from "@/components/PageShell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

type ToolCall = {
  name: string;
  args: string;
};

type TraceStep = {
  id: string;
  label: string;
  tools: ToolCall[];
  finding: string;
};

const traceSteps: TraceStep[] = [
  {
    id: "s1",
    label: "Load tool schemas",
    tools: [
      {
        name: "ToolSearch",
        args: 'select: mcp__pandects__get_server_capabilities, mcp__pandects__search_agreements, mcp__pandects__list_filter_options',
      },
    ],
    finding:
      "Schemas loaded. MCP tools cannot be called without first fetching their parameter schemas via ToolSearch.",
  },
  {
    id: "s2",
    label: "Orient to the MCP server + run initial deal search",
    tools: [
      {
        name: "mcp__pandects__get_server_capabilities",
        args: 'sections: ["concept_notes", "workflows", "field_inventory"]',
      },
      {
        name: "mcp__pandects__search_agreements",
        args: 'target_industry: ["Technology", "Software"], acquirer_pe: ["true"], transaction_price_total: ["1B - 5B"], year_min: 2021, year_max: 2026',
      },
    ],
    finding:
      'Server confirmed 5 research workflows and full field inventory. The deal search returned 0 results — "Technology" and "Software" are not valid filter values. The server uses NAICS codes for industry classification.',
  },
  {
    id: "s3",
    label: "Load more tool schemas",
    tools: [
      {
        name: "ToolSearch",
        args: 'select: mcp__pandects__search_sections, mcp__pandects__get_section_snippet, mcp__pandects__get_clause_taxonomy',
      },
    ],
    finding:
      "Schemas for search_sections, get_section_snippet, and get_clause_taxonomy loaded. These are needed to search clause text and pull snippets from individual sections.",
  },
  {
    id: "s4",
    label: "Resolve valid NAICS codes + re-run deal search",
    tools: [
      {
        name: "mcp__pandects__list_filter_options",
        args: 'fields: ["target_industries", "acquirer_industries", "acquirer_pes", "target_pes", "deal_types"]',
      },
      {
        name: "mcp__pandects__search_agreements",
        args: 'acquirer_pe: ["true"], transaction_price_total: ["1B - 5B"], year_min: 2021, year_max: 2026, page_size: 50',
      },
    ],
    finding:
      "Valid software/tech NAICS codes confirmed: 511 (software publishing), 517 (telecom), 518 (data processing), 519 (information services), 541 (professional/IT services). Broad PE + $1–5B search returned 96 deals across 2 pages — Instructure, Envestnet, Alteryx, EngageSmart, Syneos Health, Cvent, MeridianLink, Avid Technology, JAMF, AvidXchange, PROS Holdings, Couchbase, and others.",
  },
  {
    id: "s5",
    label: "Identify taxonomy node + search interim covenant sections",
    tools: [
      {
        name: "mcp__pandects__get_clause_taxonomy",
        args: "(no arguments — returns full tree)",
      },
      {
        name: "mcp__pandects__search_sections",
        args: 'standard_id: ["4703077b83eea31c"], acquirer_pe: ["true"], target_industry: ["511","517","518","519","541"], transaction_price_total: ["1B - 5B"], year_min: 2021, year_max: 2026, page_size: 50',
      },
    ],
    finding:
      'Full taxonomy tree loaded. "Covenants and Agreements → Interim Operating Covenants → Conduct of Business; Ordinary Course; Interim Restrictions" maps to id 4703077b83eea31c. Section search on that node returned 75 tagged sections across 15+ software/tech PE deals — including both affirmative obligation and forbearance covenant sections for each agreement.',
  },
  {
    id: "s6",
    label: "Pull affirmative covenant language — batch 1 (8 deals, parallel)",
    tools: [
      { name: "mcp__pandects__get_section_snippet", args: "Instructure Holdings — §5.1 Affirmative Obligations ($3.46B, Thoma Bravo, 2024)" },
      { name: "mcp__pandects__get_section_snippet", args: "Envestnet — §6.1 Conduct of the Company ($3.6B, Bain Capital, 2024)" },
      { name: "mcp__pandects__get_section_snippet", args: "Alteryx — §5.1 Affirmative Covenants ($3.49B, Clearlake + Insight, 2023)" },
      { name: "mcp__pandects__get_section_snippet", args: "EngageSmart — §5.1 Affirmative Obligations ($3.0B, Vista Equity, 2023)" },
      { name: "mcp__pandects__get_section_snippet", args: "Syneos Health — §6.01 Conduct of the Company Pending the Merger ($4.46B, Elliott consortium, 2023)" },
      { name: "mcp__pandects__get_section_snippet", args: "Cvent — §5.1 Affirmative Obligations ($4.18B, Blackstone, 2023)" },
      { name: "mcp__pandects__get_section_snippet", args: "MeridianLink — §5.1 Conduct of the Company ($1.6B, 2025)" },
      { name: "mcp__pandects__get_section_snippet", args: "Avid Technology — §5.01 Conduct of Business by the Company ($1.19B, 2023)" },
    ],
    finding:
      'Three distinct formulation patterns emerged: (1) CRE qualifier on the OC obligation — Instructure, Envestnet, Cvent; (2) RBE qualifier on the OC obligation — Alteryx, Avid Technology; (3) No efforts qualifier — EngageSmart, Syneos Health, MeridianLink. All deals include "in all material respects" regardless of pattern. Notably, Syneos and EngageSmart cabin the efforts standard exclusively to the preservation sub-obligation, leaving the OC conduct obligation as a direct covenant.',
  },
  {
    id: "s7",
    label: "Pull affirmative covenant language — batch 2 (4 deals, parallel)",
    tools: [
      { name: "mcp__pandects__get_section_snippet", args: "Everbridge — §5.1 Affirmative Obligations ($1.8B, Clearlake, 2024)" },
      { name: "mcp__pandects__get_section_snippet", args: "Integral Ad Science — §5.1 Affirmative Obligations ($1.72B, Vista Equity, 2025)" },
      { name: "mcp__pandects__get_section_snippet", args: "Couchbase — §5.1 Affirmative Covenants ($1.5B, 2025)" },
      { name: "mcp__pandects__get_section_snippet", args: "Momentive Global — §5.1 Affirmative Obligations ($1.5B, 2023)" },
    ],
    finding:
      "Pattern confirmed across 11 total deals. Everbridge (Clearlake, 2024) and Momentive (2023) carry explicit COVID-19 Measures carveouts; Couchbase (2025) and IAS (2025) do not — COVID carveouts are declining and absent from all 2025 deals in this sample. Everbridge and Couchbase use RBE; IAS uses CRE with \"in all material respects.\"",
  },
  {
    id: "s8",
    label: "Compile analysis",
    tools: [],
    finding:
      "Tallied formulation patterns across 11 deals (3 no-efforts / 4 CRE / 4 RBE). Identified strongest buyer-side precedents (Syneos $4.46B, EngageSmart $3B). Flagged material qualifiers: \"in all material respects\" near-universal; \"consistent with past practice\" present in Envestnet and MeridianLink; COVID carveouts declining post-2024. Drafted negotiating recommendation: accept \"in all material respects\" to drop the efforts qualifier entirely; anchor on Syneos Health and EngageSmart as the largest comparable deals taking that position.",
  },
];

function ToolCallRow({ tool }: { tool: ToolCall }) {
  return (
    <div className="rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-xs leading-relaxed [overflow-wrap:anywhere] [word-break:break-word]">
      <span className="text-primary">{tool.name}</span>
      <span className="text-muted-foreground">{"("}</span>
      <span className="text-foreground/80">{tool.args}</span>
      <span className="text-muted-foreground">{")"}</span>
    </div>
  );
}

export default function Examples() {
  const allStepIds = traceSteps.map((s) => s.id);

  return (
    <PageShell
      size="xl"
      title="Examples"
      subtitle="Real research sessions with the Pandects MCP server, showing the full agent trace from prompt to output."
    >
      <div className="space-y-10 overflow-x-hidden">
        {/* ── Example header ── */}
        <div className="space-y-3">
          <h2 className="text-lg font-semibold text-foreground">
            Interim Operating Covenant Market Research
          </h2>
          <div className="rounded-lg border border-border bg-muted/30 px-4 py-3">
            <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Context
            </p>
            <p className="text-sm leading-relaxed text-foreground">
              A buy-side M&A attorney uses Claude + the Pandects MCP to benchmark an interim
              operating covenant formulation against comparable PE software deals.
            </p>
            <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1.5">
              <div className="flex items-baseline gap-1.5">
                <dt className="text-xs text-muted-foreground">Date run</dt>
                <dd className="text-xs font-medium text-foreground">April 29, 2026</dd>
              </div>
              <div className="flex items-baseline gap-1.5">
                <dt className="text-xs text-muted-foreground">Model</dt>
                <dd className="text-xs font-medium text-foreground">claude-sonnet-4-6</dd>
              </div>
            </dl>
          </div>
        </div>

        {/* ── Prompt ── */}
        <section className="space-y-4" aria-labelledby="prompt-heading">
          <div className="flex items-center gap-3">
            <span aria-hidden="true" className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
              1
            </span>
            <h3 id="prompt-heading" className="text-lg font-semibold tracking-tight text-foreground whitespace-nowrap">
              Prompt
            </h3>
            <div className="ml-1 h-px flex-1 bg-border" />
            <a
              href="#output-section"
              className="whitespace-nowrap text-xs text-muted-foreground/70 transition-colors hover:text-muted-foreground"
            >
              Skip to output ↓
            </a>
          </div>
          <Card>
            <CardContent className="pt-5">
              <p className="text-sm leading-relaxed text-foreground">
                Can you run a search with the Pandects MCP for public deals in the last five years
                involving software/tech targets, PE or PE-backed buyers, and deal values around
                $1–5B?
              </p>
              <p className="mt-3 text-sm leading-relaxed text-foreground">
                Issue is the interim operating covenant. Seller wants{" "}
                <span className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                  "use commercially reasonable efforts to conduct the business in the ordinary course."
                </span>{" "}
                We want the cleaner formulation:{" "}
                <span className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                  "conduct the business in the ordinary course,"
                </span>{" "}
                without an efforts qualifier.
              </p>
              <p className="mt-3 text-sm leading-relaxed text-foreground">
                Please tell me whether our formulation is market / within range, what percentage of
                comparable deals use it or something close, and the strongest examples supporting our
                position — ideally the largest or most recognizable deals. Also flag material
                qualifiers like "in all material respects," "consistent with past practice,"
                COVID/emergency carveouts, or consent exceptions.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* ── Agent Trace ── */}
        <section className="space-y-4" aria-labelledby="trace-heading">
          <div className="flex items-center gap-3">
            <span aria-hidden="true" className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
              2
            </span>
            <h3 id="trace-heading" className="text-lg font-semibold tracking-tight text-foreground whitespace-nowrap">
              Agent Trace
            </h3>
            <div className="ml-1 h-px flex-1 bg-border" />
            <a
              href="#output-section"
              className="whitespace-nowrap text-xs text-muted-foreground/70 transition-colors hover:text-muted-foreground"
            >
              Skip to output ↓
            </a>
          </div>
          <Card>
            <CardContent className="pt-5">
              <Accordion type="multiple" defaultValue={allStepIds} className="space-y-1">
                {traceSteps.map((step, i) => (
                  <AccordionItem
                    key={step.id}
                    value={step.id}
                    className="rounded-md border border-border px-3 last:border-b"
                  >
                    <AccordionTrigger className="min-w-0 py-3 text-left hover:no-underline">
                      <div className="flex min-w-0 flex-1 items-center gap-2">
                        <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-semibold text-muted-foreground">
                          {i + 1}
                        </span>
                        <span className="truncate text-sm font-medium text-foreground">
                          {step.label}
                        </span>
                        {step.tools.length > 0 && (
                          <div className="ml-1 hidden flex-wrap gap-1 sm:flex">
                            {[...new Set(step.tools.map((t) => t.name))].slice(0, 2).map((name) => (
                              <Badge
                                key={name}
                                variant="secondary"
                                className="rounded px-1.5 py-0 font-mono text-[10px]"
                              >
                                {name.replace("mcp__pandects__", "")}
                              </Badge>
                            ))}
                            {step.tools.length > 2 && (
                              <Badge variant="secondary" className="rounded px-1.5 py-0 text-[10px]">
                                +{step.tools.length - 2}
                              </Badge>
                            )}
                          </div>
                        )}
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="pb-3">
                      <div className="space-y-3 pt-1">
                        {step.tools.length > 0 && (
                          <div className="space-y-1.5">
                            {step.tools.map((tool, j) => (
                              <ToolCallRow key={j} tool={tool} />
                            ))}
                          </div>
                        )}
                        <div className="rounded-md border-l-2 border-primary/40 bg-primary/5 px-3 py-2">
                          <p className="text-xs leading-relaxed text-foreground/80">
                            <span className="font-semibold text-foreground">Finding: </span>
                            {step.finding}
                          </p>
                        </div>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </CardContent>
          </Card>
        </section>

        {/* ── Output ── */}
        <section id="output-section" tabIndex={-1} className="space-y-4 outline-none" aria-labelledby="output-heading">
          <div className="flex items-center gap-3">
            <span aria-hidden="true" className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
              3
            </span>
            <h3 id="output-heading" className="text-lg font-semibold tracking-tight text-foreground whitespace-nowrap">
              Output
            </h3>
            <div className="ml-1 h-px flex-1 bg-border" />
          </div>
          <Card>
            <CardHeader className="pb-2 pt-5">
              <p className="text-base font-semibold text-foreground">
                Interim Operating Covenant Market Analysis
              </p>
              <p className="text-xs text-muted-foreground">
                Scope: Software/tech targets (NAICS 511, 517–519, 541), PE or PE-backed buyers,
                $1–5B total consideration, 2021–2026 — 75 tagged sections across 15+ agreements.
              </p>
            </CardHeader>
            <CardContent className="space-y-7 pb-6">

              {/* Positions */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-foreground">Your Position vs. Seller's Position</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th scope="col" className="pb-2 pr-6 text-left font-medium text-muted-foreground">Formulation</th>
                        <th scope="col" className="pb-2 text-left font-medium text-muted-foreground">Side</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      <tr>
                        <td className="py-2 pr-6 font-mono text-xs text-foreground">
                          "conduct the business in the ordinary course"
                        </td>
                        <td className="py-2">
                          <span className="inline-block rounded-md border border-border px-2 py-0.5 text-xs text-foreground">
                            Buyer's ask (clean, no qualifier)
                          </span>
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-6 font-mono text-xs text-foreground">
                          "use commercially reasonable efforts to conduct the business in the ordinary course"
                        </td>
                        <td className="py-2">
                          <span className="inline-block rounded-md border border-border px-2 py-0.5 text-xs text-foreground">
                            Seller's ask
                          </span>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Market Breakdown */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-foreground">Market Breakdown (11 deals analyzed)</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th scope="col" className="pb-2 pr-4 text-left font-medium text-muted-foreground">Formulation Structure</th>
                        <th scope="col" className="pb-2 pr-4 text-left font-medium text-muted-foreground">Count</th>
                        <th scope="col" className="pb-2 pr-4 text-left font-medium text-muted-foreground">%</th>
                        <th scope="col" className="pb-2 text-left font-medium text-muted-foreground">Deals</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border text-xs">
                      <tr>
                        <td className="py-2 pr-4 font-medium text-foreground">No efforts qualifier on OC obligation</td>
                        <td className="py-2 pr-4">3 / 11</td>
                        <td className="py-2 pr-4 font-semibold text-primary">~27%</td>
                        <td className="py-2 text-muted-foreground">EngageSmart, Syneos Health, MeridianLink</td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-4 font-medium text-foreground">CRE qualifier on OC obligation</td>
                        <td className="py-2 pr-4">4 / 11</td>
                        <td className="py-2 pr-4">~36%</td>
                        <td className="py-2 text-muted-foreground">Instructure, Envestnet, IAS, Cvent</td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-4 font-medium text-foreground">RBE qualifier on OC obligation</td>
                        <td className="py-2 pr-4">4 / 11</td>
                        <td className="py-2 pr-4">~36%</td>
                        <td className="py-2 text-muted-foreground">Alteryx, Avid, Everbridge, Couchbase</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <p className="text-xs leading-relaxed text-muted-foreground">
                  The seller's position (CRE/RBE + OC) represents ~72% of comparable deals. The
                  "no efforts" variant is clearly present in name-brand deals at scale, but is not
                  the majority practice.{" "}
                  <span className="font-medium text-foreground">
                    Critical caveat:
                  </span>{" "}
                  None of the 11 deals use the truly bare buyer formulation. Every deal — including
                  the 3 with no efforts qualifier — includes{" "}
                  <span className="font-mono text-xs">"in all material respects"</span> as a softener.
                  Your exact formulation has no direct precedent in this sample.
                </p>
              </div>

              {/* Supporting Precedents */}
              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-foreground">
                  Best Supporting Precedents for Your Position
                </h4>
                <p className="text-xs text-muted-foreground">
                  These deals drop the efforts qualifier. All retain{" "}
                  <span className="font-mono">"in all material respects."</span>
                </p>
                <div className="space-y-3">
                  {[
                    {
                      deal: "Syneos Health",
                      meta: "$4.46B · 2023 · Star Parent (Elliott-led consortium)",
                      quote:
                        "(i) conduct its operations, in all material respects, in the ordinary course of business, and (ii) use its commercially reasonable efforts to preserve the goodwill and current relationships...",
                      note:
                        "CRE is cabined to relationship preservation only — the OC conduct obligation is a standalone direct covenant. Largest deal in the no-efforts group and the most buyer-favorable structuring.",
                    },
                    {
                      deal: "EngageSmart",
                      meta: "$3.0B · 2023 · Icefall Parent (Vista Equity)",
                      quote:
                        "conduct its business and operations in the ordinary course of business in all material respects",
                      note:
                        "RBE is reserved for 'maintain existence in good standing' — a separate, weaker obligation — and OC conduct is a standalone direct covenant.",
                    },
                    {
                      deal: "MeridianLink",
                      meta: "$1.6B · 2025 · ML Holdco",
                      quote:
                        "conduct its business in all material respects in the ordinary course, consistent with past practice",
                      note:
                        "Most recent deal in the no-efforts group (Aug 2025). Note: adds 'consistent with past practice,' which can restrict new initiatives — worth pushing back on in a SaaS context.",
                    },
                  ].map(({ deal, meta, quote, note }) => (
                    <div key={deal} className="rounded-md border border-border p-4 space-y-2">
                      <div className="flex flex-wrap items-baseline gap-2">
                        <span className="font-semibold text-sm text-foreground">{deal}</span>
                        <span className="text-xs text-muted-foreground">{meta}</span>
                      </div>
                      <blockquote className="rounded bg-muted/50 px-3 py-2 font-mono text-xs leading-relaxed text-foreground/80 border-l-2 border-primary/30">
                        "{quote}"
                      </blockquote>
                      <p className="text-xs leading-relaxed text-muted-foreground">{note}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Other side */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-foreground">
                  Deals with Efforts Qualifier (Seller's Formulation or Stronger)
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-border">
                        <th scope="col" className="pb-2 pr-4 text-left font-medium text-muted-foreground">Deal</th>
                        <th scope="col" className="pb-2 pr-4 text-left font-medium text-muted-foreground">Size / Year</th>
                        <th scope="col" className="pb-2 text-left font-medium text-muted-foreground">Formulation</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {[
                        ["Instructure (Thoma Bravo)", "$3.46B / 2024", 'CRE to conduct OC "in all material respects"'],
                        ["Envestnet (Bain Capital)", "$3.6B / 2024", 'CRE to conduct OC "consistent with past practice in all material respects"'],
                        ["Cvent (Blackstone)", "$4.18B / 2023", 'CRE to conduct OC "in all material respects"'],
                        ["Alteryx (Clearlake + Insight)", "$3.49B / 2023", "RBE to conduct OC (no material respects qualifier)"],
                        ["Everbridge (Clearlake)", "$1.8B / 2024", "RBE to conduct OC"],
                        ["Couchbase", "$1.5B / 2025", "RBE to conduct OC"],
                      ].map(([deal, size, formulation]) => (
                        <tr key={deal}>
                          <td className="py-2 pr-4 font-medium text-foreground">{deal}</td>
                          <td className="py-2 pr-4 text-muted-foreground">{size}</td>
                          <td className="py-2 text-muted-foreground">{formulation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Material Qualifiers */}
              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-foreground">Material Qualifiers to Watch</h4>
                <div className="space-y-2">
                  {[
                    {
                      label: '"In all material respects"',
                      body: 'Near-universal — present in ~90%+ of deals regardless of the efforts qualifier. This is the key practical concession you should expect to make even if you win on dropping efforts. It limits the breach trigger to deviations that are material to the business overall.',
                    },
                    {
                      label: '"Consistent with past practice"',
                      body: "Present in Envestnet (CRE formulation) and MeridianLink (no-efforts). Double-edged: cabins OC to historical operations but can be used against the company if it needs to respond to market conditions. Push back in a SaaS context where GTM strategy shifts frequently.",
                    },
                    {
                      label: "COVID / Emergency Carveouts",
                      body: "Declining but not gone. Present in Everbridge (2024), Momentive (2023), and Syneos Health (2023). Absent from all 2025 deals in this sample. Safe to resist; no longer standard.",
                    },
                    {
                      label: "Consent Exception Structure",
                      body: "Universal — all deals include a written-consent carve-out (actions approved by parent, not unreasonably withheld/conditioned/delayed). Syneos and Cvent add deemed-consent timers (7 business days in Cvent). Worth negotiating a timer into the consent provision.",
                    },
                  ].map(({ label, body }) => (
                    <div key={label} className="rounded-md border border-border px-4 py-3 space-y-1">
                      <p className="text-xs font-semibold font-mono text-foreground">{label}</p>
                      <p className="text-xs leading-relaxed text-muted-foreground">{body}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Assessment */}
              <div className="rounded-md border border-primary/20 bg-primary/5 px-4 py-4 space-y-2">
                <h4 className="text-sm font-semibold text-foreground">Negotiating Assessment</h4>
                <p className="text-xs leading-relaxed text-muted-foreground">
                  Your formulation is{" "}
                  <span className="font-medium text-foreground">not currently market as written</span>{" "}
                  — the seller has the better of the pure market argument since ~73% of comparable
                  deals include an efforts qualifier. However, you have meaningful support for a
                  hybrid:{" "}
                  <span className="font-mono text-xs">"conduct the business in the ordinary course in all material respects"</span>{" "}
                  (no efforts, accept "in all material respects"). That formulation is present in
                  $3B+ deals with sophisticated PE buyers and counsel.
                </p>
                <p className="text-xs leading-relaxed text-muted-foreground">
                  <span className="font-medium text-foreground">Recommended position:</span> Push for
                  the no-efforts + "in all material respects" formulation and anchor on{" "}
                  <span className="font-medium text-foreground">Syneos Health ($4.46B, Elliott)</span>{" "}
                  and{" "}
                  <span className="font-medium text-foreground">EngageSmart ($3B, Vista Equity)</span>{" "}
                  as precedents. If the seller insists on efforts, the market-supported compromise is{" "}
                  <span className="font-medium text-foreground">CRE</span> (not RBE) — avoid RBE,
                  which is the heavier standard and less common in the CRE-heavy software buyout
                  space.
                </p>
              </div>

            </CardContent>
          </Card>
        </section>
      </div>
    </PageShell>
  );
}
