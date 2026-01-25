import { useState } from "react";
import { CircleAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { useIsMobile } from "@/hooks/use-mobile";
import { flagAsInaccurate, type FlagInaccurateSource } from "@/lib/auth-api";
import { useToast } from "@/components/ui/use-toast";

const TOOLTIP_REST =
  " Click here to flag an issue with this section or agreement; we'll look into it and correct the formatting or taxonomy classification by hand if something is amiss. (Must be signed in to use.)";

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
  const isLoggedIn = status === "authenticated";

  const payload = {
    source,
    agreementUuid,
    ...(source === "search_result" && sectionUuid != null
      ? { sectionUuid }
      : {}),
  };

  const handleClick = async () => {
    if (!isLoggedIn || submitting) return;
    setSubmitting(true);
    try {
      await flagAsInaccurate(payload);
      toast({
        title: "Flag submitted",
        description:
          "Thanks for reporting. We'll look into it and correct any issues.",
      });
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
        "h-4 w-4 shrink-0 text-red-800 dark:text-red-400",
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
      onClick={handleClick}
      disabled={submitting}
      className={cn(triggerClassName, "hover:text-foreground")}
      aria-label="Flag as inaccurate"
    >
      {icon}
    </Button>
  );

  const inactiveTrigger = isMobile ? (
    <span
      className={triggerClassName}
      role="img"
      aria-label="Flag as inaccurate (sign in to use)"
    >
      {icon}
    </span>
  ) : (
    <button
      type="button"
      className={cn(
        triggerClassName,
        "cursor-default border-none bg-transparent p-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
      )}
      aria-label="Flag as inaccurate (sign in to use)"
      aria-disabled="true"
      tabIndex={0}
      onClick={(e) => e.preventDefault()}
    >
      {icon}
    </button>
  );

  const trigger = isLoggedIn ? activeButton : inactiveTrigger;

  if (isMobile) {
    return <>{trigger}</>;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>{trigger}</TooltipTrigger>
      <TooltipContent
        side={tooltipSide}
        align={tooltipSide === "left" ? "start" : undefined}
        className="max-w-sm overflow-visible"
        sideOffset={8}
      >
        <p className="whitespace-normal break-words text-left">
          <strong>Flag as Inaccurate.</strong>
          {TOOLTIP_REST}
        </p>
      </TooltipContent>
    </Tooltip>
  );
}
