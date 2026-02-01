import { type FormEvent, useId, useState } from "react";
import { CircleAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/use-auth";
import { useIsMobile } from "@/hooks/use-mobile";
import { flagAsInaccurate, type FlagInaccurateSource } from "@/lib/auth-api";
import { useToast } from "@/components/ui/use-toast";

const TOOLTIP_REST =
  " Click here to report an issue with this section or agreement; we'll look into it and correct the formatting or taxonomy classification by hand if something is amiss.";

const ISSUE_OPTIONS = [
  "Incorrect tagging (Article/Section)",
  "Corrupted formatting",
  "Incorrect taxonomy class",
  "Incorrect metadata",
  "Not an M&A agreement",
  "Something else",
] as const;

interface FlagAsInaccurateButtonProps {
  source: FlagInaccurateSource;
  agreementUuid: string;
  sectionUuid?: string;
  className?: string;
  /** Preferred tooltip side. Use "left" when button is top-right (e.g. modal) to avoid viewport clipping. */
  tooltipSide?: "left" | "bottom";
}

export function FlagAsInaccurateButton({
  source,
  agreementUuid,
  sectionUuid,
  className,
  tooltipSide = "bottom",
}: FlagAsInaccurateButtonProps) {
  const { status } = useAuth();
  const { toast } = useToast();
  const isMobile = useIsMobile();
  const [submitting, setSubmitting] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [requestFollowUp, setRequestFollowUp] = useState(false);
  const [issueSelections, setIssueSelections] = useState<string[]>([]);
  const textareaId = useId();
  const isLoggedIn = status === "authenticated";
  const canSubmit = isLoggedIn && issueSelections.length > 0 && !submitting;
  const handleOpenChange = (nextOpen: boolean) => {
    setIsOpen(nextOpen);
    if (!nextOpen) {
      setMessage("");
      setRequestFollowUp(false);
      setIssueSelections([]);
    }
  };

  const payload = {
    source,
    agreementUuid,
    ...(source === "search_result" && sectionUuid != null
      ? { sectionUuid }
      : {}),
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      await flagAsInaccurate({
        ...payload,
        message: message.trim() || undefined,
        requestFollowUp,
        issueTypes: issueSelections,
      });
      toast({
        title: "Flag submitted",
        description:
          "Thanks for reporting. We'll look into it and correct any issues.",
      });
      setIsOpen(false);
      setMessage("");
      setRequestFollowUp(false);
      setIssueSelections([]);
    } catch (e) {
      toast({
        title: "Could not submit flag",
        description: e instanceof Error ? e.message : "Please try again later.",
        variant: "destructive",
      });
    } finally {
      setSubmitting(false);
    }
  };

  const icon = (
    <CircleAlert
      className={cn(
        "h-4 w-4 shrink-0 text-muted-foreground",
        submitting && "opacity-50",
      )}
      aria-hidden="true"
    />
  );

  const triggerClassName = cn(
    "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-muted-foreground sm:h-8 sm:w-8",
    className,
  );

  const activeButton = (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      onClick={() => setIsOpen(true)}
      disabled={submitting}
      className={cn(triggerClassName, "hover:text-foreground")}
      aria-label="Report an issue"
      aria-haspopup="dialog"
    >
      {icon}
    </Button>
  );

  const trigger = activeButton;
  const issueGroupId = `${textareaId}-issues`;

  if (isMobile) {
    return (
      <Dialog open={isOpen} onOpenChange={handleOpenChange}>
        {trigger}
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Report an issue</DialogTitle>
          </DialogHeader>
          <form className="grid gap-4" onSubmit={handleSubmit}>
            <div className="grid gap-2">
              <span id={issueGroupId} className="text-sm font-medium">
                What&apos;s wrong?{" "}
                <span className="font-normal text-muted-foreground">
                  (Required)
                </span>
              </span>
              <div
                className="grid gap-2 rounded-md border border-border/60 p-3 text-sm sm:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]"
                role="group"
                aria-labelledby={issueGroupId}
                aria-required="true"
              >
                {ISSUE_OPTIONS.map((option) => {
                  const optionId = `${textareaId}-${option.replace(/\s+/g, "-").toLowerCase()}`;
                  const isChecked = issueSelections.includes(option);
                  return (
                    <div key={option} className="flex items-center gap-2">
                      <Checkbox
                        id={optionId}
                        checked={isChecked}
                        onCheckedChange={(value) => {
                          setIssueSelections((prev) => {
                            const next = new Set(prev);
                            if (value) {
                              next.add(option);
                            } else {
                              next.delete(option);
                            }
                            return Array.from(next);
                          });
                        }}
                      />
                      <Label
                        htmlFor={optionId}
                        className={cn(
                          option === "Incorrect tagging (Article/Section)" &&
                            "whitespace-nowrap",
                        )}
                      >
                        {option}
                      </Label>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="grid gap-2">
              <span className="text-sm font-medium">
                Additional details.{" "}
                <span className="font-normal text-muted-foreground">
                  (Optional)
                </span>
              </span>
              <Label htmlFor={textareaId} className="sr-only">
                Issue description
              </Label>
              <Textarea
                id={textareaId}
                placeholder="Describe the issue..."
                value={message}
                onChange={(event) => setMessage(event.target.value)}
              />
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id={`${textareaId}-followup`}
                checked={requestFollowUp}
                onCheckedChange={(value) =>
                  setRequestFollowUp(Boolean(value))
                }
              />
              <Label htmlFor={`${textareaId}-followup`}>
                Request follow-up
              </Label>
            </div>
            <div className="flex justify-center">
              <Button type="submit" disabled={!canSubmit}>
                Submit
              </Button>
            </div>
            {!isLoggedIn && (
              <p className="text-center text-sm text-muted-foreground">
                You must be signed in to submit.
              </p>
            )}
          </form>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <Tooltip>
        <TooltipTrigger asChild>{trigger}</TooltipTrigger>
        <TooltipContent
          side={tooltipSide}
          align={tooltipSide === "left" ? "start" : undefined}
          className="max-w-sm overflow-visible"
          sideOffset={8}
        >
          <p className="whitespace-normal break-words text-left">
            <strong>Report an issue.</strong>
            {TOOLTIP_REST}
          </p>
        </TooltipContent>
      </Tooltip>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Report an issue</DialogTitle>
        </DialogHeader>
        <form className="grid gap-4" onSubmit={handleSubmit}>
          <div className="grid gap-2">
            <span id={issueGroupId} className="text-sm font-medium">
              What&apos;s wrong?{" "}
              <span className="font-normal text-muted-foreground">
                (Required)
              </span>
            </span>
            <div
              className="grid gap-2 rounded-md border border-border/60 p-3 text-sm sm:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]"
              role="group"
              aria-labelledby={issueGroupId}
              aria-required="true"
            >
              {ISSUE_OPTIONS.map((option) => {
                const optionId = `${textareaId}-${option.replace(/\s+/g, "-").toLowerCase()}`;
                const isChecked = issueSelections.includes(option);
                return (
                  <div key={option} className="flex items-center gap-2">
                    <Checkbox
                      id={optionId}
                      checked={isChecked}
                      onCheckedChange={(value) => {
                        setIssueSelections((prev) => {
                          const next = new Set(prev);
                          if (value) {
                            next.add(option);
                          } else {
                            next.delete(option);
                          }
                          return Array.from(next);
                        });
                      }}
                    />
                    <Label
                      htmlFor={optionId}
                      className={cn(
                        option === "Incorrect tagging (Article/Section)" &&
                          "whitespace-nowrap",
                      )}
                    >
                      {option}
                    </Label>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="grid gap-2">
            <span className="text-sm font-medium">
              Additional details.{" "}
              <span className="font-normal text-muted-foreground">
                (Optional)
              </span>
            </span>
            <Label htmlFor={textareaId} className="sr-only">
              Issue description
            </Label>
            <Textarea
              id={textareaId}
              placeholder="Describe the issue..."
              value={message}
              onChange={(event) => setMessage(event.target.value)}
            />
          </div>
          <div className="flex items-center gap-2">
            <Checkbox
              id={`${textareaId}-followup`}
              checked={requestFollowUp}
              onCheckedChange={(value) => setRequestFollowUp(Boolean(value))}
            />
            <Label htmlFor={`${textareaId}-followup`}>
              Request follow-up
            </Label>
          </div>
          <div className="flex justify-center">
            <Button type="submit" disabled={!canSubmit}>
              Submit
            </Button>
          </div>
          {!isLoggedIn && (
            <p className="text-center text-sm text-muted-foreground">
              You must be signed in to submit.
            </p>
          )}
        </form>
      </DialogContent>
    </Dialog>
  );
}
