import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";

type MobileChartModalProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  children: ReactNode;
};

export function MobileChartModal({
  open,
  onOpenChange,
  title,
  description,
  children,
}: MobileChartModalProps) {
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const [mounted, setMounted] = useState(false);
  const descriptionId = useId();

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!open || !mounted) return;

    const activeElement = document.activeElement as HTMLElement | null;
    const scrollY = window.scrollY;
    const body = document.body;
    const root = document.getElementById("root");
    const currentModalCount = Number(body.dataset.mobileChartModalCount ?? "0");
    const previousBodyStyle = {
      position: body.style.position,
      top: body.style.top,
      left: body.style.left,
      right: body.style.right,
      width: body.style.width,
      overflow: body.style.overflow,
    };

    body.style.position = "fixed";
    body.style.top = `-${scrollY}px`;
    body.style.left = "0";
    body.style.right = "0";
    body.style.width = "100%";
    body.style.overflow = "hidden";
    body.dataset.mobileChartModalCount = String(currentModalCount + 1);
    if (root) {
      root.setAttribute("aria-hidden", "true");
      root.setAttribute("inert", "");
    }

    const frame = window.requestAnimationFrame(() => {
      closeButtonRef.current?.focus();
    });

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onOpenChange(false);
    };

    document.addEventListener("keydown", onKeyDown);

    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", onKeyDown);
      body.style.position = previousBodyStyle.position;
      body.style.top = previousBodyStyle.top;
      body.style.left = previousBodyStyle.left;
      body.style.right = previousBodyStyle.right;
      body.style.width = previousBodyStyle.width;
      body.style.overflow = previousBodyStyle.overflow;
      const nextModalCount = Math.max(
        0,
        Number(body.dataset.mobileChartModalCount ?? "1") - 1,
      );
      body.dataset.mobileChartModalCount = String(nextModalCount);
      if (nextModalCount === 0 && root) {
        root.removeAttribute("aria-hidden");
        root.removeAttribute("inert");
      }
      window.scrollTo(0, scrollY);
      activeElement?.focus?.();
    };
  }, [mounted, onOpenChange, open]);

  if (!mounted || !open) return null;

  return createPortal(
    <div className="fixed inset-0 z-50">
      <div
        className="absolute inset-0 bg-black/80"
        onClick={() => onOpenChange(false)}
        aria-hidden="true"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        aria-describedby={description ? descriptionId : undefined}
        className="absolute inset-0 bg-background"
      >
        {description ? (
          <p id={descriptionId} className="sr-only">
            {description}
          </p>
        ) : null}
        <button
          ref={closeButtonRef}
          type="button"
          onClick={() => onOpenChange(false)}
          className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          aria-label={`Close ${title} chart`}
          style={{
            right: "max(1rem, env(safe-area-inset-right))",
            top: "max(1rem, env(safe-area-inset-top))",
          }}
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
        <div
          className="flex h-full w-full items-start justify-center overflow-y-auto p-4 [-webkit-overflow-scrolling:touch]"
          style={{
            paddingTop: "max(3.5rem, env(safe-area-inset-top))",
            paddingBottom: "max(1rem, env(safe-area-inset-bottom))",
          }}
        >
          <div className="w-full max-w-[980px]">{children}</div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
