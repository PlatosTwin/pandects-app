import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function SiteBanner() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const dismissedTimestamp = localStorage.getItem("site-banner-dismissed");
    if (!dismissedTimestamp) {
      setIsVisible(true);
    } else {
      const dismissedTime = parseInt(dismissedTimestamp);
      const now = Date.now();
      const oneDayInMs = 24 * 60 * 60 * 1000; // 24 hours in milliseconds

      if (now - dismissedTime > oneDayInMs) {
        // More than 24 hours have passed, show the banner again
        setIsVisible(true);
        localStorage.removeItem("site-banner-dismissed");
      }
    }
  }, []);

  const handleDismiss = () => {
    setIsVisible(false);
    localStorage.setItem("site-banner-dismissed", Date.now().toString());
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="bg-amber-100 border-b border-amber-300">
      <div className="max-w-9xl mx-auto px-7 py-3">
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <p className="text-sm text-amber-900">
              <span className="font-semibold">Notice</span>: This project is in
              the very early stages of its development. Site layout, API schema,
              data organization, and other features may change. Currently, we
              expose only 45 sample agreements, as a proof-of-concept.
            </p>
          </div>
          <div className="ml-4 flex-shrink-0">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleDismiss}
              className="text-amber-900 hover:text-amber-950 hover:bg-amber-200 p-1 h-auto"
            >
              <X className="w-4 h-4" />
              <span className="sr-only">Dismiss banner</span>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
