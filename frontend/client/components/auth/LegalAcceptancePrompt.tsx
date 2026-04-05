import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
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
        <div className="flex items-start gap-3 rounded-lg border border-border/60 bg-muted/20 p-4 text-sm">
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
            <Link
              to="/terms"
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:underline"
            >
              Terms of Service
            </Link>
            ,{" "}
            <Link
              to="/privacy-policy"
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:underline"
            >
              Privacy Policy
            </Link>
            , and{" "}
            <Link
              to="/license"
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:underline"
            >
              License
            </Link>
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
