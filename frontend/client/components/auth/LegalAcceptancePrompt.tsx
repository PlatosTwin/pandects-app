import { Button } from "@/components/ui/button";
import { ReferenceLink } from "@/components/ui/reference-link";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";

export function LegalAcceptancePrompt({
  email,
  checked,
  disabled,
  onCheckedChange,
  onSubmit,
  submitLabel,
}: {
  email: string;
  checked: boolean;
  disabled: boolean;
  onCheckedChange: (checked: boolean) => void;
  onSubmit: () => void;
  submitLabel: string;
}) {
  return (
    <Card className="p-6">
      <div className="grid gap-4">
        <div>
          <h2 className="text-base font-medium">One more step</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Accept the Pandects terms to finish creating or reactivating the account for{" "}
            <span className="font-medium text-foreground">{email}</span>.
          </p>
        </div>
        <div className="flex items-start gap-3 rounded-lg border border-border bg-muted/20 p-4 text-sm">
          <Checkbox
            id="legal-acceptance"
            checked={checked}
            disabled={disabled}
            onCheckedChange={(next) => onCheckedChange(next === true)}
          />
          <div className="leading-relaxed">
            <Label htmlFor="legal-acceptance" className="sr-only">
              Accept legal terms
            </Label>
            I have read and agree to the{" "}
            <ReferenceLink
              to="/terms"
              hideIcon
              className="text-primary underline underline-offset-4 hover:text-primary/80"
            >
              Terms of Service
            </ReferenceLink>
            ,{" "}
            <ReferenceLink
              to="/privacy-policy"
              hideIcon
              className="text-primary underline underline-offset-4 hover:text-primary/80"
            >
              Privacy Policy
            </ReferenceLink>
            , and{" "}
            <ReferenceLink
              to="/license"
              hideIcon
              className="text-primary underline underline-offset-4 hover:text-primary/80"
            >
              License
            </ReferenceLink>
            .
          </div>
        </div>
        <Button disabled={disabled || !checked} onClick={onSubmit} className="w-full sm:w-auto">
          {submitLabel}
        </Button>
      </div>
    </Card>
  );
}
