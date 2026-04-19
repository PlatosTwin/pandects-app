import { useMemo } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AgreementReader } from "@/components/AgreementReader";

export default function AgreementPage() {
  const { agreementUuid } = useParams<{ agreementUuid: string }>();
  const [searchParams] = useSearchParams();
  const backTo = useMemo(() => searchParams.get("from"), [searchParams]);
  const focusSectionUuid = searchParams.get("focusSectionUuid");

  if (!agreementUuid) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Agreement not found</AlertTitle>
        <AlertDescription>Missing agreement UUID in the route.</AlertDescription>
      </Alert>
    );
  }

  return (
    <AgreementReader
      agreementUuid={agreementUuid}
      focusSectionUuid={focusSectionUuid}
      backTo={backTo}
    />
  );
}
