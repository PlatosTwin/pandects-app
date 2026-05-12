import { PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import type { LucideIcon } from "lucide-react";
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

const helpCardClass =
  "rounded-lg border border-border bg-card p-4 transition-colors hover:border-primary/40 hover:bg-accent/40 block";

function HelpCard({
  icon: Icon,
  title,
  description,
  href,
  to,
  external,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  href?: string;
  to?: string;
  external?: boolean;
}) {
  const body = (
    <>
      <Icon className="h-5 w-5 text-primary mb-2" aria-hidden />
      <h3 className="text-base font-medium text-foreground">
        {title}
        {external && (
          <ExternalLink
            className="ml-0.5 inline-block h-3 w-3 align-baseline opacity-60"
            aria-hidden
          />
        )}
      </h3>
      <p className="text-sm text-muted-foreground mt-1">{description}</p>
    </>
  );

  if (href) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={external ? `${title} (opens in a new tab)` : title}
        className={helpCardClass}
      >
        {body}
      </a>
    );
  }
  return (
    <Link to={to!} className={helpCardClass}>
      {body}
    </Link>
  );
}

function SectionHeader({
  number,
  title,
  id,
}: {
  number: string;
  title: string;
  id: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 text-primary text-xs font-medium tabular-nums">
        {number}
      </span>
      <h2
        id={id}
        className="text-2xl font-semibold tracking-tight text-foreground"
      >
        {title}
      </h2>
    </div>
  );
}

export default function Support() {
  return (
    <PageShell size="lg" title="Support">
      <article>
        <section
          id="support-development"
          className="scroll-mt-24 first:pt-0"
          aria-labelledby="support-development-heading"
        >
          <SectionHeader
            number="01"
            title="Support development"
            id="support-development-heading"
          />

          <p className="prose-copy mt-4">
            Pandects is free to use, but the infrastructure behind the corpus
            has ongoing costs. Contributions help keep the site reliable,
            current, and open.
          </p>

          <div className="mt-6 rounded-lg border border-border bg-card p-4 shadow-sm sm:p-5">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div className="max-w-2xl">
                <div className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-muted-foreground">
                  <Heart className="h-4 w-4 text-primary" aria-hidden="true" />
                  Buy Me a Coffee
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  A lightweight way to help cover operating expenses and
                  ongoing development.
                </p>
              </div>
              <Button asChild size="lg" className="w-fit gap-2 px-8">
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
            number="02"
            title="Operating costs"
            id="operating-costs-heading"
          />
          <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3">
            {moneyItems.map(({ icon: Icon, title, description }) => (
              <div
                key={title}
                className="rounded-lg border border-border bg-card p-4"
              >
                <Icon
                  className="h-5 w-5 text-primary mb-2"
                  aria-hidden="true"
                />
                <h3 className="text-base font-medium text-foreground">
                  {title}
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  {description}
                </p>
              </div>
            ))}
          </div>
        </section>

        <section
          id="other-ways"
          className="scroll-mt-24 pt-12"
          aria-labelledby="other-ways-heading"
        >
          <SectionHeader
            number="03"
            title="Other ways to help"
            id="other-ways-heading"
          />
          <div className="mx-auto mt-6 grid max-w-3xl grid-cols-1 gap-3 md:grid-cols-2">
            <HelpCard
              icon={Star}
              title="Star on GitHub"
              description="Help the project's visibility."
              href="https://github.com/PlatosTwin/pandects-app"
              external
            />
            <HelpCard
              icon={MessageSquareWarning}
              title="Open an issue"
              description="Report a bug or request a feature."
              href="https://github.com/PlatosTwin/pandects-app/issues"
              external
            />
          </div>
        </section>
      </article>
    </PageShell>
  );
}
