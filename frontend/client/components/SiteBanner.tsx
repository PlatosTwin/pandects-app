import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function SiteBanner() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const dismissed = localStorage.getItem("site-banner-dismissed");
    if (!dismissed) {
      setIsVisible(true);
    }
  }, []);

  const handleDismiss = () => {
    setIsVisible(false);
    localStorage.setItem("site-banner-dismissed", "true");
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="bg-blue-50 border-b border-blue-200">
      <div className="max-w-9xl mx-auto px-7 py-3">
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <p className="text-sm text-blue-900">
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
              className="text-amber-800 hover:text-amber-900 hover:bg-amber-100 p-1 h-auto"
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
