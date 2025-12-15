import { useLLMOutput } from "@/hooks/use-llm-output";
import { cn } from "@/lib/utils";
import { PageShell } from "@/components/PageShell";
import {
  Search,
  Check,
  Settings,
  Copy,
  Trash2,
  Minus,
  Plus,
  FileText,
} from "lucide-react";
import SaveConfirmationModal from "@/components/SaveConfirmationModal";
import ErrorModal from "@/components/ErrorModal";
import InfoModal from "@/components/InfoModal";

export default function Edit() {
  const { state, actions } = useLLMOutput();

  return (
    <div className="w-full flex flex-col">
      <PageShell
        size="md"
        title="LLM Output Editor"
        subtitle="Load and edit LLM-generated content with live preview and save functionality."
        className="max-w-4xl px-8"
      >
        <div className="flex flex-col gap-8">

        {/* Header Section */}
        <div className="flex items-start gap-10 flex-wrap">
          {/* Page UUID Input */}
          <div className="flex flex-col gap-6 flex-1 min-w-[300px]">
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-normal leading-5 tracking-[0.15px] text-muted-foreground">
                Page UUID
              </label>
              <div className="relative flex items-center">
                <input
                  type="text"
                  placeholder="Enter page UUID"
                  value={state.pageUuid}
                  onChange={(e) => actions.updatePageUuid(e.target.value)}
                  onKeyDown={(e) => {
                    if (
                      e.key === "Enter" &&
                      !state.isLoading &&
                      state.pageUuid.trim()
                    ) {
                      actions.loadPage();
                    }
                  }}
                  tabIndex={1}
                  className="flex-1 text-base font-normal leading-6 tracking-[0.15px] text-foreground bg-transparent border-none min-h-6 py-1 focus:outline-none focus:bg-accent transition-colors"
                />
                <div className="absolute bottom-0 left-0 right-0 h-px bg-border" />
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-4">
            {/* Load Page Button */}
            <button
              type="button"
              disabled={state.isLoading || !state.pageUuid.trim()}
              onClick={actions.loadPage}
              tabIndex={2}
              style={{ opacity: state.isLoading ? 0.7 : 1 }}
              className={cn(
                "flex items-center justify-center gap-2 px-6 py-2 rounded-md bg-primary text-primary-foreground text-[15px] font-medium leading-[26px] tracking-[0.46px] uppercase transition-all duration-200",
                "shadow-[0px_1px_5px_0px_rgba(0,0,0,0.12),0px_2px_2px_0px_rgba(0,0,0,0.14),0px_3px_1px_-2px_rgba(0,0,0,0.20)]",
                "hover:shadow-[0px_2px_8px_0px_rgba(0,0,0,0.15),0px_3px_4px_0px_rgba(0,0,0,0.18),0px_4px_2px_-2px_rgba(0,0,0,0.25)]",
                "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
                "disabled:opacity-50 disabled:cursor-not-allowed",
              )}
            >
              <Search
                className={cn(
                  "w-6 h-6 flex-shrink-0",
                  state.isLoading && "animate-spin-custom",
                )}
              />
              <span>{state.isLoading ? "Loading..." : "Load page"}</span>
            </button>

            {/* Save Button */}
            <button
              type="button"
              disabled={state.isLoading}
              onClick={actions.requestSave}
              tabIndex={3}
              className="flex items-center justify-center gap-2 px-6 py-2 rounded-md border border-primary/40 text-primary text-[15px] font-medium leading-[26px] tracking-[0.46px] uppercase transition-all duration-200 hover:bg-primary/10 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            >
              <Check className="w-6 h-6 flex-shrink-0" />
              <span>Save</span>
            </button>

            {/* Settings Button */}
            <button
              type="button"
              onClick={actions.toggleSettings}
              className="flex items-center justify-center p-2 rounded-md border border-input text-muted-foreground transition-all duration-200 hover:bg-accent"
              aria-label="Toggle settings"
            >
              <Settings className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        {state.showSettings && (
          <div className="flex items-center gap-4 p-4 bg-muted rounded-md border-l-4 border-primary">
            {/* Font Size Controls */}
            <div className="flex items-center gap-2">
              <label className="text-xs font-normal text-muted-foreground">
                Font size:
              </label>
              <button
                type="button"
                onClick={actions.decreaseFontSize}
                className="w-8 h-8 flex items-center justify-center rounded border border-input text-muted-foreground hover:bg-accent transition-all duration-200"
              >
                <Minus className="w-4 h-4" />
              </button>
              <span className="text-sm font-normal text-foreground min-w-8 text-center">
                {state.fontSize}px
              </span>
              <button
                type="button"
                onClick={actions.increaseFontSize}
                className="w-8 h-8 flex items-center justify-center rounded border border-input text-muted-foreground hover:bg-accent transition-all duration-200"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Divider */}
        <div className="w-full h-px bg-border" />

        {/* Content Section */}
        <div className="flex flex-col gap-4 flex-1">
          {/* Content Header */}
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              <h2 className="text-base font-medium text-foreground">
                LLM Output
              </h2>
              {state.lastSaved && (
                <div className="text-xs text-muted-foreground flex items-center leading-none">
                  <span>Last saved: </span>
                  <span>{state.lastSaved}</span>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2">
              {/* Stats */}
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span>{state.wordCount} words</span>
                <span>{state.charCount} characters</span>
              </div>

              {/* Copy Button */}
              <button
                type="button"
                onClick={actions.copyToClipboard}
                className="flex items-center gap-1 px-3 py-1 rounded border border-input text-xs text-muted-foreground transition-all duration-200 hover:bg-accent"
              >
                <Copy className="w-4 h-4" />
                <span>Copy</span>
              </button>

              {/* Clear Button */}
              <button
                type="button"
                onClick={actions.clearContent}
                className="flex items-center gap-1 px-3 py-1 rounded border border-input text-xs text-muted-foreground transition-all duration-200 hover:bg-accent"
              >
                <Trash2 className="w-4 h-4" />
                <span>Clear</span>
              </button>
            </div>
          </div>

          {/* Text Area Container */}
          <div className="relative flex-1 min-h-[400px]">
            <div className="absolute inset-0 rounded-t-md bg-muted p-3">
              <div className="flex flex-col gap-3 h-full">
                <label className="text-xs font-normal leading-5 tracking-[0.15px] text-muted-foreground">
                  LLM Output
                </label>
                <textarea
                  className="resize-none flex-1 w-full bg-transparent border-none text-foreground font-normal leading-6 tracking-[0.15px] focus:outline-none"
                  placeholder="Enter or load content here..."
                  value={state.llmOutput}
                  onChange={(e) => actions.updateLLMOutput(e.target.value)}
                  style={{ fontSize: `${state.fontSize}px` }}
                />
              </div>
            </div>
            <div className="absolute bottom-0 left-0 right-0 h-px bg-border" />
          </div>
        </div>
        </div>
      </PageShell>

      {/* Copy Success Toast */}
      {state.showCopySuccess && (
        <div className="fixed bottom-6 right-6 bg-primary text-primary-foreground px-4 py-2 rounded shadow-[0_10px_15px_-3px_rgb(0_0_0_/_0.1),0_4px_6px_-4px_rgb(0_0_0_/_0.1)] text-sm">
          Copied to clipboard!
        </div>
      )}

      {/* Copy Error Toast */}
      {state.copyError && (
        <div className="fixed bottom-6 right-6 bg-red-600 text-white px-4 py-2 rounded shadow-[0_10px_15px_-3px_rgb(0_0_0_/_0.1),0_4px_6px_-4px_rgb(0_0_0_/_0.1)] text-sm">
          Copy failed. Please select and copy manually.
        </div>
      )}

      {/* Save Success Toast */}
      {state.showSaveSuccess && (
        <div className="fixed bottom-6 right-6 bg-green-600 text-white px-4 py-2 rounded shadow-[0_10px_15px_-3px_rgb(0_0_0_/_0.1),0_4px_6px_-4px_rgb(0_0_0_/_0.1)] text-sm">
          Changes saved successfully!
        </div>
      )}

      {/* Save Confirmation Modal */}
      <SaveConfirmationModal
        isOpen={state.showSaveConfirmation}
        onConfirm={actions.confirmSave}
        onCancel={actions.cancelSave}
        title="Confirm Save"
        message="Are you sure you want to save these changes? This will overwrite any existing content."
      />

      {/* Error Modal */}
      <ErrorModal
        isOpen={state.showErrorModal}
        onClose={actions.closeErrorModal}
        message={state.errorMessage}
      />

      {/* Info Modal */}
      <InfoModal
        isOpen={state.showInfoModal}
        onClose={actions.closeInfoModal}
        title="Page Not Found"
        message={state.infoMessage}
      />
    </div>
  );
}
