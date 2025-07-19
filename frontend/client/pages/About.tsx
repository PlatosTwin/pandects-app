import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import Navigation from "@/components/Navigation";

export default function About() {
  const [activeSection, setActiveSection] = useState("");

  useEffect(() => {
    // Handle hash fragment on page load
    const hash = window.location.hash.slice(1); // Remove the '#'
    if (hash) {
      // Small delay to ensure the page has rendered
      setTimeout(() => {
        scrollToSection(hash);
        setActiveSection(hash);
      }, 100);
    }
  }, []);

  const scrollToSection = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
    setActiveSection(id);
  };

  return (
    <div className="min-h-screen bg-cream">
      {/* Header */}
      <Navigation />

      <div className="flex">
        {/* Sidebar */}
        <nav className="hidden md:block w-64 flex-shrink-0 border-r border-material-divider">
          <div className="px-6 pt-6 pb-4">
            <ul className="space-y-2">
              {[
                { id: "overview", label: "Overview" },
                { id: "data", label: "Data" },
                { id: "sources", label: "Sources", indent: true },
                {
                  id: "processing-pipelines",
                  label: "Processing pipelines",
                  indent: true,
                },
                { id: "contributing", label: "Contributing" },
                { id: "credits", label: "Credits" },
              ].map(({ id, label, indent }) => (
                <li key={id} className={indent ? "ml-4" : undefined}>
                  <button
                    onClick={() => scrollToSection(id)}
                    className={cn(
                      "w-full text-left px-3 py-2 text-sm rounded-md transition-colors",
                      activeSection === id
                        ? "bg-material-blue-light text-material-blue font-medium"
                        : "text-material-text-secondary hover:text-material-text-primary hover:bg-material-surface",
                    )}
                  >
                    {label}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 px-8 py-8">
          <div className="max-w-4xl mx-auto space-y-6">
            <section id="overview" className="scroll-mt-8">
              <h1 className="text-3xl font-bold text-material-text-primary mb-6">
                Overview
              </h1>
              <div>{/* empty—no box */}</div>
            </section>

            <section id="data" className="scroll-mt-8">
              <h2 className="text-2xl font-bold text-material-text-primary mb-4">
                Data
              </h2>
              <div></div>

              <div id="sources" className="scroll-mt-8 mt-12">
                <h3 className="text-xl font-semibold text-material-text-primary mb-4">
                  Sources
                </h3>
                <div></div>
              </div>

              <div id="processing-pipelines" className="scroll-mt-8 mt-12">
                <h3 className="text-xl font-semibold text-material-text-primary mb-4">
                  Processing pipelines
                </h3>
                <div></div>
              </div>
            </section>

            <section id="contributing" className="scroll-mt-8">
              <h2 className="text-2xl font-bold text-material-text-primary mb-4">
                Contributing
              </h2>
              <div>
                This is an open-source project, and we welcome contributions.
                Please see the{" "}
                <a
                  href="https://github.com/PlatosTwin/pandects-app"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-material-blue hover:underline"
                >
                  Github repository
                </a>{" "}
                for more on contributing.
              </div>
            </section>

            <section id="credits" className="scroll-mt-8">
              <h2 className="text-2xl font-bold text-material-text-primary mb-4">
                Credits
              </h2>
              <div className="flex flex-col space-y-6">
                <p>
                  Professor Emiliano Catan at NYU Law has been an active advisor
                  to this project from the beginning. We are also thankful to
                  NYU Law professor Chris Sprigman. The concept and design for
                  this site borrows heavily from the wonderful{" "}
                  <a
                    href="https://case.law"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-material-blue hover:underline"
                  >
                    Caselaw Access Project
                  </a>
                  , a product of the Library Innovation Lab (LIL) at Harvard
                  Law; Jack Cushman at LIL provided guidance and advice early on
                  in this project. Josh Carty provided technical assistance
                  early on, and helped brainstorm.
                </p>
                <p>
                  This project would not have gotten off the ground without the
                  prior work of Peter Adelson, Matthew Jennejohn, Julian Nyarko,
                  and Eric Talley, whose{" "}
                  <a
                    href="https://onlinelibrary.wiley.com/doi/abs/10.1111/jels.12410"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-material-blue hover:underline"
                  >
                    <em>
                      Introducing a New Corpus of Definitive M&A Agreements,
                      2000–2020
                    </em>{" "}
                  </a>
                  provided the inspiration and seed set of source links to
                  definitive agreements in EDGAR. Their Github repository is
                  available{" "}
                  <a
                    href="https://github.com/padelson/dma_corpus/tree/main"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-material-blue hover:underline"
                  >
                    here
                  </a>
                  .
                </p>
              </div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
