import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface SaveConfirmationModalProps {
  isOpen: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  title?: string;
  message?: string;
}

export default function SaveConfirmationModal({
  isOpen,
  onConfirm,
  onCancel,
  title = "Confirm Save",
  message = "Are you sure you want to save these changes?",
}: SaveConfirmationModalProps) {
  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) onCancel();
      }}
    >
      <DialogContent
        className="sm:max-w-md"
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            e.stopPropagation();
            onConfirm();
          }
        }}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <AlertTriangle className="h-5 w-5 text-amber-500" />
            {title}
          </DialogTitle>
          {message && <DialogDescription>{message}</DialogDescription>}
        </DialogHeader>

        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button onClick={onConfirm}>Save Changes</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
