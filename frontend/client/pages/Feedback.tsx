import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { PageShell } from "@/components/PageShell";
import { trackEvent } from "@/lib/analytics";
import { useEffect, useMemo, useState } from "react";

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
  const [loadTimeouts, setLoadTimeouts] = useState<
    Record<"survey" | "general-feedback", boolean>
  >({
    survey: false,
    "general-feedback": false,
  });

  useEffect(() => {
    if (typeof document === "undefined") return;
    if (document.querySelector("script[data-airtable-embed]")) return;

    const loadScript = () => {
      const script = document.createElement("script");
      script.async = true;
      script.src = "https://static.airtable.com/js/embed/embed_snippet_v1.js";
      script.dataset.airtableEmbed = "true";
      document.head.appendChild(script);
    };

    if (window.requestIdleCallback) {
      window.requestIdleCallback(loadScript);
    } else {
      window.setTimeout(loadScript, 500);
    }
  }, []);

  const handleSectionChange = (value: string) => {
    setOpenSection(value);
    if (value === "survey" || value === "general-feedback") {
      setMountedSections((prev) => ({ ...prev, [value]: true }));
    }
  };

  useEffect(() => {
    if (!mountedSections.survey || loadedSections.survey) return;
    const timer = window.setTimeout(
      () => setLoadTimeouts((prev) => ({ ...prev, survey: true })),
      10000,
    );
    return () => window.clearTimeout(timer);
  }, [mountedSections.survey, loadedSections.survey]);

  useEffect(() => {
    if (!mountedSections["general-feedback"] || loadedSections["general-feedback"]) return;
    const timer = window.setTimeout(
      () => setLoadTimeouts((prev) => ({ ...prev, "general-feedback": true })),
      10000,
    );
    return () => window.clearTimeout(timer);
  }, [mountedSections["general-feedback"], loadedSections["general-feedback"]]);

  const surveyUrl =
    "https://airtable.com/embed/appsaasOdbK3k0JIR/pagNFOMrP8gZLyEl3/form";
  const generalFeedbackUrl =
    "https://airtable.com/embed/appsaasOdbK3k0JIR/pagX6sJC7D7wihUto/form";

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
            We're collecting feedback from end users on the data and tooling.
            How do you plan to access Pandects data? Do you have comments on the
            proposed XML schema or taxonomy? Anything else we should consider?
            Let us know by submitting the survey below.
          </p>
          <p>
            We also have a general feedback form for issues, questions, or
            improvements. You can also{" "}
            <a
              href="https://github.com/PlatosTwin/pandects-app/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-2 hover:underline"
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
              className="text-primary underline underline-offset-2 hover:underline"
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
            headingLevel="h2"
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
                    src={surveyUrl}
                    title="Pandects user survey form"
                    width="100%"
                    height="895"
                    onLoad={() =>
                      setLoadedSections((prev) => ({ ...prev, survey: true }))
                    }
                  />
                  {loadTimeouts.survey && !loadedSections.survey && (
                    <div className="mt-4 rounded-lg border border-border bg-background/60 p-4 text-sm text-muted-foreground">
                      The survey is taking longer than expected to load.{" "}
                      <a
                        href={surveyUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary underline underline-offset-2 hover:underline"
                      >
                        Open the survey in a new tab
                      </a>
                      .
                    </div>
                  )}
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
            headingLevel="h2"
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
                    src={generalFeedbackUrl}
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
                  {loadTimeouts["general-feedback"] &&
                    !loadedSections["general-feedback"] && (
                      <div className="mt-4 rounded-lg border border-border bg-background/60 p-4 text-sm text-muted-foreground">
                        The form is taking longer than expected to load.{" "}
                        <a
                          href={generalFeedbackUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline underline-offset-2 hover:underline"
                        >
                          Open the form in a new tab
                        </a>
                        .
                      </div>
                    )}
                </div>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </PageShell>
  );
}
