import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function SiteBanner() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    try {
      const dismissedTimestamp = localStorage.getItem("site-banner-dismissed");
      if (!dismissedTimestamp) {
        setIsVisible(true);
        return;
      }

      const dismissedTime = parseInt(dismissedTimestamp, 10);
      if (!Number.isFinite(dismissedTime)) {
        setIsVisible(true);
        localStorage.removeItem("site-banner-dismissed");
        return;
      }

      const now = Date.now();
      const oneDayInMs = 24 * 60 * 60 * 1000;
      if (now - dismissedTime > oneDayInMs) {
        setIsVisible(true);
        localStorage.removeItem("site-banner-dismissed");
      }
    } catch {
      setIsVisible(true);
    }
  }, []);

  const handleDismiss = () => {
    setIsVisible(false);
    try {
      localStorage.setItem("site-banner-dismissed", Date.now().toString());
    } catch {
      // ignore
    }
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="hidden border-b border-border bg-muted/40 md:block">
      <div className="mx-auto max-w-7xl px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1 min-w-0">
            <p className="text-sm text-muted-foreground text-justify sm:text-left">
              <span className="font-semibold text-foreground">Notice</span>:{" "}
              Pandects is in early development. Layout, API schema, and data
              organization may change. Currently, the public site includes 45
              sample agreements as a proof of concept.
            </p>
          </div>
          <div className="ml-4 flex-shrink-0">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleDismiss}
              className="p-1 h-auto text-muted-foreground hover:text-foreground"
            >
              <X className="w-4 h-4" aria-hidden="true" />
              <span className="sr-only">Dismiss banner</span>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
