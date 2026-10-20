import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

const BETA_DISCLAIMER =
  "Pandects is in early development. Layout, API schema, and data organization may change.";

export function NavBetaBadge({ className }: { className?: string }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <AlertDialog open={isOpen} onOpenChange={setIsOpen}>
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className={cn(
          "relative rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground ring-1 ring-inset ring-border before:absolute before:-inset-x-2 before:-inset-y-3 before:content-[''] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          className,
        )}
        aria-label="Read beta notice"
      >
        BETA
      </button>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Beta Notice</AlertDialogTitle>
          <AlertDialogDescription className="text-left">
            {BETA_DISCLAIMER}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogAction onClick={() => setIsOpen(false)}>Got it</AlertDialogAction>
      </AlertDialogContent>
    </AlertDialog>
  );
}
