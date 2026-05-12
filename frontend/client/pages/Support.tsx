import type { ComponentType } from "react";
import { PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
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
  BookOpen,
} from "lucide-react";
import { trackEvent } from "@/lib/analytics";

const moneyItems = [
  {
    icon: Server,
    title: "Hosting & compute",
    description: "API servers, MCP server, prerender.",
  },
  {
    icon: Database,
    title: "Database & backups",
    description: "MariaDB primary, daily snapshots, R2 archives.",
  },
  {
    icon: Workflow,
    title: "Pipeline runs",
    description: "Monthly ingest, ML tagging, embeddings.",
  },
  {
    icon: GitBranch,
    title: "CI/CD",
    description: "Test, build, and deploy automation.",
  },
];

type IconType = ComponentType<{ className?: string; "aria-hidden"?: boolean }>;

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
  icon: IconType;
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

function SectionHeader({ number, title }: { number: string; title: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 text-primary text-xs font-medium tabular-nums">
        {number}
      </span>
      <h2 className="text-2xl font-semibold tracking-tight text-foreground">
        {title}
      </h2>
    </div>
  );
}

export default function Support() {
  return (
    <PageShell size="xl" title="Support">
      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
        PROJECT · SUPPORT
      </div>
      <p className="mt-3 text-lg text-muted-foreground max-w-[68ch]">
        Pandects is free to use. If it's useful to you, here are ways to help
        keep it that way.
      </p>
      <div className="mt-6 h-px w-12 bg-primary/70" />

      <div className="mt-10 rounded-xl border border-border bg-card p-6 lg:p-8 flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <Heart className="h-4 w-4 text-primary" />
          <span className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
            Support development
          </span>
        </div>
        <h2 className="text-2xl font-semibold tracking-tight text-foreground">
          Buy me a coffee
        </h2>
        <p className="text-sm text-muted-foreground max-w-[60ch]">
          Hosting, compute, backups, and pipeline runs aren't free. Every
          contribution helps keep the site fast, reliable, and open.
        </p>
        <div className="self-start">
          <Button asChild size="lg" className="gap-2 px-8">
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
              <ExternalLink
                className="ml-0.5 inline-block h-3 w-3 opacity-60"
                aria-hidden="true"
              />
            </a>
          </Button>
        </div>
        <p className="text-xs italic text-muted-foreground">
          Pandects is not a registered nonprofit, so contributions are not
          tax-deductible.
        </p>
      </div>

      <section className="pt-12">
        <SectionHeader number="01" title="Where the money goes" />
        <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3">
          {moneyItems.map(({ icon: Icon, title, description }) => (
            <div
              key={title}
              className="rounded-lg border border-border bg-card p-4"
            >
              <Icon className="h-5 w-5 text-primary mb-2" aria-hidden="true" />
              <h3 className="text-base font-medium text-foreground">{title}</h3>
              <p className="text-sm text-muted-foreground mt-1">
                {description}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="pt-12">
        <SectionHeader number="02" title="Other ways to help" />
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3">
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
          <HelpCard
            icon={BookOpen}
            title="Cite Pandects"
            description="Acknowledge the project in your work."
            to="/about#credits"
          />
        </div>
      </section>
    </PageShell>
  );
}
