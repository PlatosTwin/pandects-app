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
      'Server confirmed 5 research workflows and full field inventory. The deal search returned 0 results - "Technology" and "Software" are not valid filter values. The server uses NAICS codes for industry classification.',
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
      "Valid software/tech NAICS codes confirmed: 511 (software publishing), 517 (telecom), 518 (data processing), 519 (information services), 541 (professional/IT services). Broad PE + $1-5B search returned 96 deals across 2 pages - Instructure, Envestnet, Alteryx, EngageSmart, Syneos Health, Cvent, MeridianLink, Avid Technology, JAMF, AvidXchange, PROS Holdings, Couchbase, and others.",
  },
  {
    id: "s5",
    label: "Identify taxonomy node + search interim covenant sections",
    tools: [
      {
        name: "mcp__pandects__get_clause_taxonomy",
        args: "(no arguments - returns full tree)",
      },
      {
        name: "mcp__pandects__search_sections",
        args: 'standard_id: ["4703077b83eea31c"], acquirer_pe: ["true"], target_industry: ["511","517","518","519","541"], transaction_price_total: ["1B - 5B"], year_min: 2021, year_max: 2026, page_size: 50',
      },
    ],
    finding:
      'Full taxonomy tree loaded. "Covenants and Agreements -> Interim Operating Covenants -> Conduct of Business; Ordinary Course; Interim Restrictions" maps to id 4703077b83eea31c. Section search on that node returned 75 tagged sections across 15+ software/tech PE deals - including both affirmative obligation and forbearance covenant sections for each agreement.',
  },
  {
    id: "s6",
    label: "Pull affirmative covenant language - batch 1 (8 deals, parallel)",
    tools: [
      { name: "mcp__pandects__get_section_snippet", args: "Instructure Holdings - §5.1 Affirmative Obligations ($3.46B, Thoma Bravo, 2024)" },
      { name: "mcp__pandects__get_section_snippet", args: "Envestnet - §6.1 Conduct of the Company ($3.6B, Bain Capital, 2024)" },
      { name: "mcp__pandects__get_section_snippet", args: "Alteryx - §5.1 Affirmative Covenants ($3.49B, Clearlake + Insight, 2023)" },
      { name: "mcp__pandects__get_section_snippet", args: "EngageSmart - §5.1 Affirmative Obligations ($3.0B, Vista Equity, 2023)" },
      { name: "mcp__pandects__get_section_snippet", args: "Syneos Health - §6.01 Conduct of the Company Pending the Merger ($4.46B, Elliott consortium, 2023)" },
      { name: "mcp__pandects__get_section_snippet", args: "Cvent - §5.1 Affirmative Obligations ($4.18B, Blackstone, 2023)" },
      { name: "mcp__pandects__get_section_snippet", args: "MeridianLink - §5.1 Conduct of the Company ($1.6B, 2025)" },
      { name: "mcp__pandects__get_section_snippet", args: "Avid Technology - §5.01 Conduct of Business by the Company ($1.19B, 2023)" },
    ],
    finding:
      'Three distinct formulation patterns emerged: (1) CRE qualifier on the OC obligation - Instructure, Envestnet, Cvent; (2) RBE qualifier on the OC obligation - Alteryx, Avid Technology; (3) No efforts qualifier - EngageSmart, Syneos Health, MeridianLink. All deals include "in all material respects" regardless of pattern. Notably, Syneos and EngageSmart cabin the efforts standard exclusively to the preservation sub-obligation, leaving the OC conduct obligation as a direct covenant.',
  },
  {
    id: "s7",
    label: "Pull affirmative covenant language - batch 2 (4 deals, parallel)",
    tools: [
      { name: "mcp__pandects__get_section_snippet", args: "Everbridge - §5.1 Affirmative Obligations ($1.8B, Clearlake, 2024)" },
      { name: "mcp__pandects__get_section_snippet", args: "Integral Ad Science - §5.1 Affirmative Obligations ($1.72B, Vista Equity, 2025)" },
      { name: "mcp__pandects__get_section_snippet", args: "Couchbase - §5.1 Affirmative Covenants ($1.5B, 2025)" },
      { name: "mcp__pandects__get_section_snippet", args: "Momentive Global - §5.1 Affirmative Obligations ($1.5B, 2023)" },
    ],
    finding:
      'Pattern confirmed across 11 total deals. Everbridge (Clearlake, 2024) and Momentive (2023) carry explicit COVID-19 Measures carveouts; Couchbase (2025) and IAS (2025) do not - COVID carveouts are declining and absent from all 2025 deals in this sample. Everbridge and Couchbase use RBE; IAS uses CRE with "in all material respects."',
  },
  {
    id: "s8",
    label: "Compile analysis",
    tools: [],
    finding:
      'Tallied formulation patterns across 11 deals (3 no-efforts / 4 CRE / 4 RBE). Identified strongest buyer-side precedents (Syneos $4.46B, EngageSmart $3B). Flagged material qualifiers: "in all material respects" near-universal; "consistent with past practice" present in Envestnet and MeridianLink; COVID carveouts declining post-2024. Drafted negotiating recommendation: accept "in all material respects" to drop the efforts qualifier entirely; anchor on Syneos Health and EngageSmart as the largest comparable deals taking that position.',
  },
];

const supportingPrecedents = [
  {
    deal: "Syneos Health",
    meta: "$4.46B · 2023 · Star Parent (Elliott-led consortium)",
    quote:
      "(i) conduct its operations, in all material respects, in the ordinary course of business, and (ii) use its commercially reasonable efforts to preserve the goodwill and current relationships...",
    note:
      "CRE is cabined to relationship preservation only - the OC conduct obligation is a standalone direct covenant. Largest deal in the no-efforts group and the most buyer-favorable structuring.",
  },
  {
    deal: "EngageSmart",
    meta: "$3.0B · 2023 · Icefall Parent (Vista Equity)",
    quote:
      "conduct its business and operations in the ordinary course of business in all material respects",
    note:
      "RBE is reserved for 'maintain existence in good standing' - a separate, weaker obligation - and OC conduct is a standalone direct covenant.",
  },
  {
    deal: "MeridianLink",
    meta: "$1.6B · 2025 · ML Holdco",
    quote:
      "conduct its business in all material respects in the ordinary course, consistent with past practice",
    note:
      "Most recent deal in the no-efforts group (Aug 2025). Note: adds 'consistent with past practice,' which can restrict new initiatives - worth pushing back on in a SaaS context.",
  },
];

const effortsDeals = [
  ["Instructure (Thoma Bravo)", "$3.46B / 2024", 'CRE to conduct OC "in all material respects"'],
  ["Envestnet (Bain Capital)", "$3.6B / 2024", 'CRE to conduct OC "consistent with past practice in all material respects"'],
  ["Cvent (Blackstone)", "$4.18B / 2023", 'CRE to conduct OC "in all material respects"'],
  ["Alteryx (Clearlake + Insight)", "$3.49B / 2023", "RBE to conduct OC (no material respects qualifier)"],
  ["Everbridge (Clearlake)", "$1.8B / 2024", "RBE to conduct OC"],
  ["Couchbase", "$1.5B / 2025", "RBE to conduct OC"],
];

const materialQualifiers = [
  {
    label: '"In all material respects"',
    body: 'Near-universal - present in ~90%+ of deals regardless of the efforts qualifier. This is the key practical concession you should expect to make even if you win on dropping efforts. It limits the breach trigger to deviations that are material to the business overall.',
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
    body: "Universal - all deals include a written-consent carve-out (actions approved by parent, not unreasonably withheld/conditioned/delayed). Syneos and Cvent add deemed-consent timers (7 business days in Cvent). Worth negotiating a timer into the consent provision.",
  },
];

function ToolCallRow({ tool }: { tool: ToolCall }) {
  return (
    <div className="research-examples__tool-call">
      <span className="research-examples__tool-name">{tool.name}</span>
      <span className="research-examples__muted">(</span>
      <span>{tool.args}</span>
      <span className="research-examples__muted">)</span>
    </div>
  );
}

function SectionHeading({
  number,
  title,
  headingId,
  link,
}: {
  number: number;
  title: string;
  headingId?: string;
  link?: string;
}) {
  return (
    <div className="research-examples__section-heading">
      <span className="research-examples__section-number" aria-hidden="true">
        {number}
      </span>
      <h2 id={headingId}>{title}</h2>
      <div className="research-examples__rule" />
      {link ? (
        <a className="research-examples__skip-link" href={link}>
          Skip to output ↓
        </a>
      ) : null}
    </div>
  );
}

export function ResearchExamplesIntro() {
  return (
    <div className="research-examples__intro">
      <div className="research-examples__context">
        <p className="research-examples__eyebrow">Context</p>
        <p>
          A buy-side M&A attorney uses Claude + the Pandects MCP to benchmark an interim operating
          covenant formulation against comparable PE software deals.
        </p>
        <dl>
          <div>
            <dt>Date run</dt>
            <dd>April 29, 2026</dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>claude-sonnet-4-6</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}

export function ResearchExamplesPrompt() {
  return (
      <section>
        <SectionHeading
          number={1}
          title="Prompt"
          link="#output"
        />
        <div className="research-examples__card">
          <p>
            Can you run a search with the Pandects MCP for public deals in the last five years
            involving software/tech targets, PE or PE-backed buyers, and deal values around $1-5B?
          </p>
          <p>
            Issue is the interim operating covenant. Seller wants{" "}
            <code>"use commercially reasonable efforts to conduct the business in the ordinary course."</code>{" "}
            We want the cleaner formulation: <code>"conduct the business in the ordinary course,"</code>{" "}
            without an efforts qualifier.
          </p>
          <p>
            Please tell me whether our formulation is market / within range, what percentage of
            comparable deals use it or something close, and the strongest examples supporting our
            position - ideally the largest or most recognizable deals. Also flag material qualifiers
            like "in all material respects," "consistent with past practice," COVID/emergency
            carveouts, or consent exceptions.
          </p>
        </div>
      </section>
  );
}

export function ResearchExamplesTrace() {
  return (
      <section>
        <SectionHeading
          number={2}
          title="Agent Trace"
          link="#output"
        />
        <div className="research-examples__card">
          <div className="research-examples__trace">
            {traceSteps.map((step, i) => (
              <details key={step.id} className="research-examples__trace-step" open>
                <summary>
                  <span className="research-examples__trace-index">{i + 1}</span>
                  <span>{step.label}</span>
                  {step.tools.length > 0 ? (
                    <span className="research-examples__badges">
                      {[...new Set(step.tools.map((t) => t.name))].slice(0, 2).map((name) => (
                        <span key={name} className="research-examples__badge">
                          {name.replace("mcp__pandects__", "")}
                        </span>
                      ))}
                      {step.tools.length > 2 ? (
                        <span className="research-examples__badge">+{step.tools.length - 2}</span>
                      ) : null}
                    </span>
                  ) : null}
                </summary>
                <div className="research-examples__trace-body">
                  {step.tools.length > 0 ? (
                    <div className="research-examples__tool-list">
                      {step.tools.map((tool, j) => (
                        <ToolCallRow key={`${step.id}-${j}`} tool={tool} />
                      ))}
                    </div>
                  ) : null}
                  <div className="research-examples__finding">
                    <strong>Finding: </strong>
                    {step.finding}
                  </div>
                </div>
              </details>
            ))}
          </div>
        </div>
      </section>
  );
}

export function ResearchExamplesOutput() {
  return (
      <section className="research-examples__output" tabIndex={-1}>
        <SectionHeading number={3} title="Output" />
        <div className="research-examples__card">
          <div className="research-examples__card-header">
            <h3>Interim Operating Covenant Market Analysis</h3>
            <p>
              Scope: Software/tech targets (NAICS 511, 517-519, 541), PE or PE-backed buyers,
              $1-5B total consideration, 2021-2026 - 75 tagged sections across 15+ agreements.
            </p>
          </div>

          <div className="research-examples__output-body">
            <div>
              <h4>Your Position vs. Seller's Position</h4>
              <div className="research-examples__table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Formulation</th>
                      <th>Side</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>
                        <code>"conduct the business in the ordinary course"</code>
                      </td>
                      <td>
                        <span className="research-examples__pill">Buyer's ask (clean, no qualifier)</span>
                      </td>
                    </tr>
                    <tr>
                      <td>
                        <code>"use commercially reasonable efforts to conduct the business in the ordinary course"</code>
                      </td>
                      <td>
                        <span className="research-examples__pill">Seller's ask</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h4>Market Breakdown (11 deals analyzed)</h4>
              <div className="research-examples__table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Formulation Structure</th>
                      <th>Count</th>
                      <th>%</th>
                      <th>Deals</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>No efforts qualifier on OC obligation</td>
                      <td>3 / 11</td>
                      <td>
                        <strong>~27%</strong>
                      </td>
                      <td>EngageSmart, Syneos Health, MeridianLink</td>
                    </tr>
                    <tr>
                      <td>CRE qualifier on OC obligation</td>
                      <td>4 / 11</td>
                      <td>~36%</td>
                      <td>Instructure, Envestnet, IAS, Cvent</td>
                    </tr>
                    <tr>
                      <td>RBE qualifier on OC obligation</td>
                      <td>4 / 11</td>
                      <td>~36%</td>
                      <td>Alteryx, Avid, Everbridge, Couchbase</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="research-examples__note">
                The seller's position (CRE/RBE + OC) represents ~72% of comparable deals. The "no
                efforts" variant is clearly present in name-brand deals at scale, but is not the
                majority practice. <strong>Critical caveat:</strong> None of the 11 deals use the
                truly bare buyer formulation. Every deal - including the 3 with no efforts qualifier
                - includes <code>"in all material respects"</code> as a softener. Your exact
                formulation has no direct precedent in this sample.
              </p>
            </div>

            <div>
              <h4>Best Supporting Precedents for Your Position</h4>
              <p className="research-examples__note">
                These deals drop the efforts qualifier. All retain <code>"in all material respects."</code>
              </p>
              <div className="research-examples__precedents">
                {supportingPrecedents.map(({ deal, meta, quote, note }) => (
                  <div key={deal} className="research-examples__precedent">
                    <div className="research-examples__precedent-title">
                      <strong>{deal}</strong>
                      <span>{meta}</span>
                    </div>
                    <blockquote>"{quote}"</blockquote>
                    <p>{note}</p>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4>Deals with Efforts Qualifier (Seller's Formulation or Stronger)</h4>
              <div className="research-examples__table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Deal</th>
                      <th>Size / Year</th>
                      <th>Formulation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {effortsDeals.map(([deal, size, formulation]) => (
                      <tr key={deal}>
                        <td>{deal}</td>
                        <td>{size}</td>
                        <td>{formulation}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h4>Material Qualifiers to Watch</h4>
              <div className="research-examples__qualifiers">
                {materialQualifiers.map(({ label, body }) => (
                  <div key={label} className="research-examples__qualifier">
                    <p>{label}</p>
                    <span>{body}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="research-examples__assessment">
              <h4>Negotiating Assessment</h4>
              <p>
                Your formulation is <strong>not currently market as written</strong> - the seller
                has the better of the pure market argument since ~73% of comparable deals include an
                efforts qualifier. However, you have meaningful support for a hybrid:{" "}
                <code>"conduct the business in the ordinary course in all material respects"</code>{" "}
                (no efforts, accept "in all material respects"). That formulation is present in
                $3B+ deals with sophisticated PE buyers and counsel.
              </p>
              <p>
                <strong>Recommended position:</strong> Push for the no-efforts + "in all material
                respects" formulation and anchor on <strong>Syneos Health ($4.46B, Elliott)</strong>{" "}
                and <strong>EngageSmart ($3B, Vista Equity)</strong> as precedents. If the seller
                insists on efforts, the market-supported compromise is <strong>CRE</strong> (not RBE)
                - avoid RBE, which is the heavier standard and less common in the CRE-heavy software
                buyout space.
              </p>
            </div>
          </div>
        </div>
      </section>
  );
}

export default function ResearchExamples() {
  return (
    <div className="research-examples">
      <ResearchExamplesIntro />
      <ResearchExamplesPrompt />
      <ResearchExamplesTrace />
      <ResearchExamplesOutput />
    </div>
  );
}
