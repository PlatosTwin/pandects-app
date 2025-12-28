import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { PageShell } from "@/components/PageShell";
import { trackEvent } from "@/lib/analytics";
import { useMemo, useState } from "react";

export default function Feedback() {
  const [openSection, setOpenSection] = useState<string>("");
  const [mountedSections, setMountedSections] = useState<
    Record<"survey" | "general-feedback", boolean>
  >({
    survey: false,
    "general-feedback": false,
  });
  const [loadedSections, setLoadedSections] = useState<
    Record<"survey" | "general-feedback", boolean>
  >({
    survey: false,
    "general-feedback": false,
  });

  const handleSectionChange = (value: string) => {
    setOpenSection(value);
    if (value === "survey" || value === "general-feedback") {
      setMountedSections((prev) => ({ ...prev, [value]: true }));
    }
  };

  const skeletonBySection = useMemo(
    () => ({
      survey: (
        <div
          className="h-[895px] rounded-lg border border-border bg-muted/30 p-6"
          role="status"
          aria-live="polite"
        >
          <span className="sr-only">Loading survey form</span>
          <div className="h-4 w-40 animate-pulse rounded bg-muted" />
          <div className="mt-4 h-3 w-full animate-pulse rounded bg-muted" />
          <div className="mt-2 h-3 w-5/6 animate-pulse rounded bg-muted" />
          <div className="mt-2 h-3 w-2/3 animate-pulse rounded bg-muted" />
        </div>
      ),
      "general-feedback": (
        <div
          className="h-[895px] rounded-lg border border-border bg-muted/30 p-6"
          role="status"
          aria-live="polite"
        >
          <span className="sr-only">Loading feedback form</span>
          <div className="h-4 w-48 animate-pulse rounded bg-muted" />
          <div className="mt-4 h-3 w-full animate-pulse rounded bg-muted" />
          <div className="mt-2 h-3 w-5/6 animate-pulse rounded bg-muted" />
          <div className="mt-2 h-3 w-2/3 animate-pulse rounded bg-muted" />
        </div>
      ),
    }),
    [],
  );

  return (
    <PageShell size="xl" title="Feedback">
      <div className="mb-8">
        <div className="prose max-w-none text-muted-foreground space-y-4 mb-8">
          <p>
            We're currently soliciting input from our end users on all things
            data. How are you planning to access Pandects data? Do you have
            comments on the proposed XML schema or the taxonomy? Are there
            other things we should be taking into consideration? Let us know by
            submitting the survey form below!
          </p>
          <p>
            We also have a form for general feedback, where you can flag
            issues, submit questions, or propose improvements. Alternatively,
            you can{" "}
            <a
              href="https://github.com/PlatosTwin/pandects-app/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
              aria-label="Open an issue (opens in a new tab)"
            >
              open an issue
            </a>{" "}
            on GitHub or be the change you want to see and submit a pull
            request. For more on contributing, see the{" "}
            <a
              href="https://github.com/PlatosTwin/pandects-app"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
              aria-label="Main GitHub repository (opens in a new tab)"
            >
              main GitHub repository
            </a>
            .
          </p>
          <p className="font-medium text-foreground">
            Thanks for helping make Pandects better!
          </p>
        </div>
      </div>

      <Accordion
        type="single"
        collapsible
        value={openSection}
        onValueChange={handleSectionChange}
        className="w-full space-y-4"
      >
        <AccordionItem
          value="survey"
          className="bg-card rounded-lg shadow-sm border border-border"
        >
          <AccordionTrigger
            className="px-6 py-4 text-lg font-semibold text-foreground"
            onClick={() =>
              trackEvent("feedback_section_click", { section: "survey" })
            }
          >
            Survey
          </AccordionTrigger>
          <AccordionContent className="px-6 pb-6">
            <div className="bg-card rounded-lg">
              {!mountedSections.survey ? (
                skeletonBySection.survey
              ) : (
                <div
                  className="relative"
                  aria-busy={!loadedSections.survey}
                >
                  {!loadedSections.survey && (
                    <div className="absolute inset-0" aria-hidden="true">
                      {skeletonBySection.survey}
                    </div>
                  )}
                  <iframe
                    loading="lazy"
                    className="airtable-embed w-full rounded-lg"
                    style={{
                      background: "transparent",
                      border: "1px solid #ccc",
                      opacity: loadedSections.survey ? 1 : 0,
                    }}
                    src="https://airtable.com/embed/appsaasOdbK3k0JIR/pagNFOMrP8gZLyEl3/form"
                    title="Pandects user survey form"
                    width="100%"
                    height="895"
                    onLoad={() =>
                      setLoadedSections((prev) => ({ ...prev, survey: true }))
                    }
                  />
                </div>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem
          value="general-feedback"
          className="bg-card rounded-lg shadow-sm border border-border"
        >
          <AccordionTrigger
            className="px-6 py-4 text-lg font-semibold text-foreground"
            onClick={() =>
              trackEvent("feedback_section_click", {
                section: "general_feedback",
              })
            }
          >
            General Feedback
          </AccordionTrigger>
          <AccordionContent className="px-6 pb-6">
            <div className="bg-card rounded-lg">
              {!mountedSections["general-feedback"] ? (
                skeletonBySection["general-feedback"]
              ) : (
                <div
                  className="relative"
                  aria-busy={!loadedSections["general-feedback"]}
                >
                  {!loadedSections["general-feedback"] && (
                    <div className="absolute inset-0" aria-hidden="true">
                      {skeletonBySection["general-feedback"]}
                    </div>
                  )}
                  <iframe
                    loading="lazy"
                    className="airtable-embed w-full rounded-lg"
                    style={{
                      background: "transparent",
                      border: "1px solid #ccc",
                      opacity: loadedSections["general-feedback"] ? 1 : 0,
                    }}
                    src="https://airtable.com/embed/appsaasOdbK3k0JIR/pagX6sJC7D7wihUto/form"
                    title="Pandects general feedback form"
                    width="100%"
                    height="895"
                    onLoad={() =>
                      setLoadedSections((prev) => ({
                        ...prev,
                        "general-feedback": true,
                      }))
                    }
                  />
                </div>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </PageShell>
  );
}
