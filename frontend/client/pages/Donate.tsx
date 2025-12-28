import { PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Coffee } from "lucide-react";
import { trackEvent } from "@/lib/analytics";

export default function Donate() {
  return (
    <PageShell
      size="xl"
      title="Donate"
      subtitle="Help keep the project running."
    >
      <div className="mb-8">
        <div className="prose max-w-none text-muted-foreground space-y-4">
          <p>
            Pandects is an open-source project with real, ongoing costs: hosting
            and compute for the API, database storage, backups, CI/CD, and the
            pipeline runs that ingest and process new agreements.
          </p>
          <p>
            If you find the project useful and you want to support development
            and operating expenses, you can contribute via Buy Me a Coffee.
            Every contribution helps keep the site fast, reliable, and free to
            use.
          </p>
          <p className="italic">
            Disclaimer: Although the navigation says "Donate," Pandects is not
            a registered nonprofit, so contributions via Buy Me a Coffee are
            not tax-deductible.
          </p>
        </div>
      </div>

      <div className="flex flex-col items-center gap-2">
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
          </a>
        </Button>
        <div className="text-xs text-muted-foreground">Opens in a new tab.</div>
      </div>
    </PageShell>
  );
}
