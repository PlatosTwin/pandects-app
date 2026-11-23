import { lazy, Suspense, useState } from "react";
import { cn } from "@/lib/utils";

// The dialog loads on the first click, so Radix AlertDialog stays out of the
// entry bundle.
const NavBetaNotice = lazy(() => import("@/components/NavBetaNotice"));

export function NavBetaBadge({ className }: { className?: string }) {
  const [isOpen, setIsOpen] = useState(false);
  // Stays true after the first click, so the dialog stays mounted and can
  // play its close animation.
  const [hasOpened, setHasOpened] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => {
          setHasOpened(true);
          setIsOpen(true);
        }}
        className={cn(
          "relative rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground ring-1 ring-inset ring-border before:absolute before:-inset-x-2 before:-inset-y-3 before:content-[''] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          className,
        )}
        aria-label="Read beta notice"
      >
        BETA
      </button>
      {hasOpened && (
        <Suspense fallback={null}>
          <NavBetaNotice open={isOpen} onOpenChange={setIsOpen} />
        </Suspense>
      )}
    </>
  );
}
