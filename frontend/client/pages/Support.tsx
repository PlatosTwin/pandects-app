import { PageShell } from "@/components/PageShell";
import { SectionHeader } from "@/components/SectionHeader";
import { Button } from "@/components/ui/button";
import { FactCard } from "@/components/FactCard";
import { Eyebrow } from "@/components/ui/eyebrow";
import {
  Heart,
  Coffee,
  ExternalLink,
  Server,
  Database,
  Workflow,
  GitBranch,
  Star,
  MessageSquareWarning,
} from "lucide-react";
import { trackEvent } from "@/lib/analytics";

const moneyItems = [
  {
    icon: Server,
    title: "Fly.io hosting",
    description: "Frontend, API, MCP server, and auth database.",
  },
  {
    icon: Database,
    title: "Database & backups",
    description: "MariaDB primary, daily snapshots, R2 archives.",
  },
  {
    icon: Workflow,
    title: "OpenAI API",
    description: "ETL classification, extraction, and enrichment calls.",
  },
  {
    icon: GitBranch,
    title: "CI/CD",
    description: "Test, build, and deploy automation.",
  },
];

export default function Support() {
  return (
    <PageShell layout="docs" title="Support">
      <article>
        <section
          id="support-development"
          className="scroll-mt-24"
          aria-labelledby="support-development-heading"
        >
          <SectionHeader
            title="Support development"
            id="support-development-heading"
          />

          <p className="prose-copy mt-4">
            Pandects is free to use, but the infrastructure behind the corpus
            has ongoing costs. Contributions help keep the site reliable,
            current, and open.
          </p>

          <div className="mt-6 rounded-xl border border-border bg-card shadow-raised p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <Eyebrow className="flex items-center gap-2">
                  <Heart className="h-4 w-4 text-primary" aria-hidden="true" />
                  Buy Me a Coffee
                </Eyebrow>
                <p className="mt-2 text-sm text-muted-foreground">
                  A lightweight way to help cover operating expenses and
                  ongoing development.
                </p>
              </div>
              <Button asChild size="lg" className="w-fit shrink-0 gap-2 px-6">
                <a
                  href="https://www.buymeacoffee.com/nmbogdan"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Buy Me a Coffee (opens in a new tab)"
                  onClick={() =>
                    trackEvent("buy_me_a_coffee_click", {
                      href: "https://www.buymeacoffee.com/nmbogdan",
                    })
                  }
                >
                  <Coffee className="h-4 w-4" aria-hidden="true" />
                  Buy Me a Coffee
                  <ExternalLink className="h-3 w-3" aria-hidden="true" />
                </a>
              </Button>
            </div>
            <p className="mt-4 text-xs italic text-muted-foreground">
              Pandects is not a registered nonprofit, so contributions are not
              tax-deductible.
            </p>
          </div>
        </section>

        <section
          id="operating-costs"
          className="scroll-mt-24 pt-12"
          aria-labelledby="operating-costs-heading"
        >
          <SectionHeader
            title="Operating costs"
            id="operating-costs-heading"
          />
          <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3">
            {moneyItems.map(({ icon, title, description }) => (
              <FactCard
                key={title}
                icon={icon}
                title={title}
                description={description}
              />
            ))}
          </div>
        </section>

        <section
          id="other-ways"
          className="scroll-mt-24 pt-12"
          aria-labelledby="other-ways-heading"
        >
          <SectionHeader
            title="Other ways to help"
            id="other-ways-heading"
          />
          <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <FactCard
              icon={Star}
              title="Star on GitHub"
              description="Help the project's visibility."
              href="https://github.com/PlatosTwin/pandects-app"
            />
            <FactCard
              icon={MessageSquareWarning}
              title="Open an issue"
              description="Report a bug or request a feature."
              href="https://github.com/PlatosTwin/pandects-app/issues"
            />
          </div>
        </section>
      </article>
    </PageShell>
  );
}
