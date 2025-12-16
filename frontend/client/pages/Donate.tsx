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
        <div className="prose text-muted-foreground space-y-4">
          <p>
            Pandects is an open-source project with real, ongoing costs: hosting
            and compute for the API, database storage, backups, CI/CD, and the
            pipeline runs that ingest and process new agreements.
          </p>
          <p>
            If you find the project useful and you’d like to support development
            and operating expenses, you can contribute via Buy Me A Coffee. 
            Every contribution helps keep the site fast, reliable, and free to
            use.
          </p>
        </div>
      </div>

      <div className="flex flex-col items-center gap-2">
        <Button asChild size="lg" className="gap-2 px-8">
          <a
            href="https://www.buymeacoffee.com/nmbogdan"
            target="_blank"
            rel="noopener noreferrer"
            onClick={() =>
              trackEvent("buy_me_a_coffee_click", {
                href: "https://www.buymeacoffee.com/nmbogdan",
              })
            }
          >
            <Coffee className="h-4 w-4" />
            Buy me a coffee
          </a>
        </Button>
        <div className="text-xs text-muted-foreground">Opens in a new tab.</div>
      </div>
    </PageShell>
  );
}
