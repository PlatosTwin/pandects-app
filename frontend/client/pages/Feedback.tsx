import { FactCard } from "@/components/FactCard";
import { PageShell } from "@/components/PageShell";
import { SectionHeader } from "@/components/SectionHeader";
import { trackEvent } from "@/lib/analytics";
import { useEffect, useRef } from "react";
import { Bug, GitPullRequest, MessageSquare } from "lucide-react";

const generalFeedbackUrl =
  "https://airtable.com/embed/appsaasOdbK3k0JIR/pagX6sJC7D7wihUto/form";

export default function Feedback() {
  const trackedRef = useRef(false);

  useEffect(() => {
    if (trackedRef.current) return;
    trackedRef.current = true;
    trackEvent("feedback_form_view", { section: "general_feedback" });
  }, []);

  return (
    <PageShell layout="docs" title="Feedback">
      <section
        id="channels"
        className="scroll-mt-24"
        aria-labelledby="channels-heading"
      >
        <SectionHeader title="Channels" id="channels-heading" />

        <p className="prose-copy mt-4">
          Questions, bug reports, and feature requests are welcome. Choose the
          path that matches the shape of the feedback.
        </p>

        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          <FactCard
            icon={MessageSquare}
            eyebrow="Feedback"
            title="Use the form"
            description="Best for quick notes, questions, and product suggestions."
            href={generalFeedbackUrl}
          />
          <FactCard
            icon={Bug}
            eyebrow="GitHub"
            title="Open an issue"
            description="Best for reproducible bugs and scoped feature requests."
            href="https://github.com/PlatosTwin/pandects-app/issues"
          />
          <FactCard
            icon={GitPullRequest}
            eyebrow="Contributions"
            title="Open a PR"
            description="Best when the fix or improvement is already in code."
            href="https://github.com/PlatosTwin/pandects-app/compare"
          />
        </div>
      </section>
    </PageShell>
  );
}
