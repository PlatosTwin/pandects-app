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

/**
 * The dialog behind the BETA badge. NavBetaBadge loads this module on the
 * first click, so Radix AlertDialog stays out of the entry bundle.
 */
export default function NavBetaNotice({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Beta Notice</AlertDialogTitle>
          <AlertDialogDescription className="text-left">
            {BETA_DISCLAIMER}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogAction onClick={() => onOpenChange(false)}>
          Got it
        </AlertDialogAction>
      </AlertDialogContent>
    </AlertDialog>
  );
}
