import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { PageShell } from "@/components/PageShell";

export default function Feedback() {
  return (
    <PageShell size="md" title="Feedback">
      <div className="mb-8">
        <div className="prose text-muted-foreground space-y-4 mb-8">
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
            >
              main GitHub repository
            </a>
            .
          </p>
          <p className="font-medium text-foreground">
            Thanks for helping to make Pandects better!
          </p>
        </div>
      </div>

      <Accordion type="single" collapsible className="w-full space-y-4">
        <AccordionItem
          value="survey"
          className="bg-card rounded-lg shadow-sm border border-border"
        >
          <AccordionTrigger className="px-6 py-4 text-lg font-semibold text-foreground">
            Survey
          </AccordionTrigger>
          <AccordionContent className="px-6 pb-6">
            <div className="bg-card rounded-lg">
              <iframe
                loading="lazy"
                className="airtable-embed w-full rounded-lg"
                src="https://airtable.com/embed/appsaasOdbK3k0JIR/pagNFOMrP8gZLyEl3/form"
                width="100%"
                height="895"
                style={{
                  background: "transparent",
                  border: "1px solid #ccc",
                }}
              />
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem
          value="general-feedback"
          className="bg-card rounded-lg shadow-sm border border-border"
        >
          <AccordionTrigger className="px-6 py-4 text-lg font-semibold text-foreground">
            General Feedback
          </AccordionTrigger>
          <AccordionContent className="px-6 pb-6">
            <div className="bg-card rounded-lg">
              <iframe
                loading="lazy"
                className="airtable-embed w-full rounded-lg"
                src="https://airtable.com/embed/appsaasOdbK3k0JIR/pagX6sJC7D7wihUto/form"
                width="100%"
                height="895"
                style={{
                  background: "transparent",
                  border: "1px solid #ccc",
                }}
              />
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </PageShell>
  );
}
