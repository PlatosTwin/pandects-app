import { ExternalLink } from "@/components/ui/external-link";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface ReaderDetailItem {
  label: string;
  value: string;
}

/** Full deal metadata, shown in the Details popover (desktop) or sheet (mobile). */
export function ReaderDetails({
  metadata,
  filingUrl,
}: {
  metadata: ReaderDetailItem[];
  filingUrl?: string | null;
}) {
  return (
    <div className="space-y-3">
      <dl className="divide-y divide-border/50">
        {metadata.map((item) => (
          <div
            key={item.label}
            className="grid grid-cols-[120px_minmax(0,1fr)] gap-3 py-1.5 text-sm"
          >
            <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              {item.label}
            </dt>
            <dd className="break-words text-foreground">{item.value}</dd>
          </div>
        ))}
      </dl>
      {filingUrl ? (
        <ExternalLink
          href={filingUrl}
          className={cn(buttonVariants({ variant: "outline", size: "sm" }), "w-full justify-center")}
        >
          View SEC filing
        </ExternalLink>
      ) : null}
    </div>
  );
}
