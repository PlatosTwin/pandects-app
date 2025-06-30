import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { Info, X } from "lucide-react";

interface InfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  message: string;
}

export default function InfoModal({
  isOpen,
  onClose,
  title = "Information",
  message,
}: InfoModalProps) {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === "Escape") {
          e.preventDefault();
          e.stopPropagation();
          onClose();
        }
      }}
      tabIndex={-1}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black bg-opacity-50"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white rounded-lg shadow-xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <Info className="w-6 h-6 text-material-blue" />
            <h3 className="text-lg font-medium text-material-text-primary">
              {title}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          <p className="text-material-text-secondary">{message}</p>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-200">
          <button
            onClick={onClose}
            className={cn(
              "px-4 py-2 text-sm font-medium text-white bg-material-blue rounded-md transition-colors",
              "hover:bg-blue-700 focus:ring-2 focus:ring-material-blue focus:ring-offset-2",
            )}
          >
            OK
          </button>
        </div>
      </div>
    </div>
  );
}
