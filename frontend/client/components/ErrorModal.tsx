import { AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface ErrorModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  message: string;
}

export default function ErrorModal({
  isOpen,
  onClose,
  title = "Error",
  message,
}: ErrorModalProps) {
  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent
        className="sm:max-w-md"
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            e.stopPropagation();
            onClose();
          }
        }}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-destructive" />
            {title}
          </DialogTitle>
          <DialogDescription>{message}</DialogDescription>
        </DialogHeader>

        <DialogFooter>
          <Button variant="destructive" onClick={onClose}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
