import { Link, useParams, useSearchParams } from "react-router-dom";
import { AgreementReader } from "@/components/AgreementReader";
import { ErrorState } from "@/components/StateMessage";
import { Button } from "@/components/ui/button";

/**
 * `/agreements/:agreementUuid?focusSectionUuid=&from=` — see
 * `lib/agreement-links.ts` for the query contract and `buildAgreementHref`.
 */
export default function AgreementPage() {
  const { agreementUuid } = useParams<{ agreementUuid: string }>();
  const [searchParams] = useSearchParams();
  const backTo = searchParams.get("from");
  const focusSectionUuid = searchParams.get("focusSectionUuid");

  if (!agreementUuid) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-10 sm:px-6 lg:px-8">
        <ErrorState
          error={{ status: 404 }}
          title="Agreement not found"
          body="This link is missing the agreement id."
          action={
            <Button asChild variant="outline" size="sm">
              <Link to="/search">Go to Search</Link>
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <AgreementReader
      // Remount per agreement so per-document state (find, matches, scroll
      // position bookkeeping) never leaks between agreements.
      key={agreementUuid}
      agreementUuid={agreementUuid}
      focusSectionUuid={focusSectionUuid}
      backTo={backTo}
    />
  );
}
