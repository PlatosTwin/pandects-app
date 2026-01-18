import { useEffect, useState, type ReactNode, type ComponentProps } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

interface AdaptiveTooltipProps {
  trigger: ReactNode;
  content: ReactNode;
  tooltipProps?: ComponentProps<typeof TooltipContent>;
  popoverProps?: ComponentProps<typeof PopoverContent>;
  /**
   * Force a specific mode instead of auto-detecting.
   * Useful when you want consistent behavior regardless of device.
   */
  forceMode?: "tooltip" | "popover";
  /**
   * Delay duration for tooltip (only applies to Tooltip, not Popover).
   * Note: This requires TooltipProvider to have delayDuration set.
   * For per-tooltip delays, wrap in a TooltipProvider with delayDuration.
   */
  delayDuration?: number;
}

/**
 * AdaptiveTooltip automatically switches between Tooltip (for precise pointers)
 * and Popover (for coarse pointers like touch) based on device capabilities.
 * This provides better UX across different input methods.
 */
export function AdaptiveTooltip({
  trigger,
  content,
  tooltipProps,
  popoverProps,
  forceMode,
  delayDuration,
}: AdaptiveTooltipProps) {
  const [isCoarsePointer, setIsCoarsePointer] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined" || forceMode) return;
    const media = window.matchMedia("(pointer: coarse)");
    const update = () => setIsCoarsePointer(media.matches);
    update();
    if (media.addEventListener) {
      media.addEventListener("change", update);
      return () => media.removeEventListener("change", update);
    }
    media.addListener(update);
    return () => media.removeListener(update);
  }, [forceMode]);

  const usePopover = forceMode === "popover" || (!forceMode && isCoarsePointer);

  if (usePopover) {
    return (
      <Popover>
        <PopoverTrigger asChild>{trigger}</PopoverTrigger>
        <PopoverContent {...popoverProps}>{content}</PopoverContent>
      </Popover>
    );
  }

  return (
    <Tooltip {...(delayDuration !== undefined ? { delayDuration } : {})}>
      <TooltipTrigger asChild>{trigger}</TooltipTrigger>
      <TooltipContent {...tooltipProps}>{content}</TooltipContent>
    </Tooltip>
  );
}
