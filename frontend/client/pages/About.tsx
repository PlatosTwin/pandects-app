import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import Navigation from "@/components/Navigation";

export default function About() {
  const [activeSection, setActiveSection] = useState("overview");

  useEffect(() => {
    const handleScroll = () => {
      const sections = [
        "overview",
        "data",
        "sources",
        "processing-pipelines",
        "contributing",
        "credits",
      ];
      const scrollY = window.scrollY + 150;

      if (
        window.innerHeight + window.scrollY >=
        document.body.offsetHeight - 100
      ) {
        setActiveSection("credits");
        return;
      }

      let current = "overview";
      for (let i = sections.length - 1; i >= 0; i--) {
        const el = document.getElementById(sections[i]);
        if (el && scrollY >= el.offsetTop) {
          current = sections[i];
          break;
        }
      }
      setActiveSection(current);
    };

    // Handle hash fragment on page load
    const hash = window.location.hash.slice(1); // Remove the '#'
    if (hash) {
      // Small delay to ensure the page has rendered
      setTimeout(() => {
        scrollToSection(hash);
      }, 100);
    }

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-cream">
      {/* Header */}
      <Navigation />

      <div className="flex">
        {/* Sidebar */}
        <nav className="hidden md:block w-64 flex-shrink-0 border-r border-material-divider">
          <div
            className="sticky top-16 px-6 pt-6 pb-4"
            style={{
              maxHeight: "calc(100vh - 8rem)", // subtract header (4rem) + footer (4rem)
              overflowY: "auto",
            }}
          >
            <ul className="space-y-4">
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
          <div className="max-w-4xl mx-auto space-y-16">
            <section id="overview" className="scroll-mt-8">
              <h1 className="text-3xl font-bold text-material-text-primary mb-6">
                Overview
              </h1>
              <div className="min-h-[400px]">{/* empty—no box */}</div>
            </section>

            <section id="data" className="scroll-mt-8">
              <h2 className="text-2xl font-bold text-material-text-primary mb-6">
                Data
              </h2>
              <div className="min-h-[300px]"></div>

              <div id="sources" className="scroll-mt-8 mt-12">
                <h3 className="text-xl font-semibold text-material-text-primary mb-4">
                  Sources
                </h3>
                <div className="min-h-[300px]"></div>
              </div>

              <div id="processing-pipelines" className="scroll-mt-8 mt-12">
                <h3 className="text-xl font-semibold text-material-text-primary mb-4">
                  Processing pipelines
                </h3>
                <div className="min-h-[300px]"></div>
              </div>
            </section>

            <section id="contributing" className="scroll-mt-8">
              <h2 className="text-2xl font-bold text-material-text-primary mb-6">
                Contributing
              </h2>
              <div className="min-h-[400px]"></div>
            </section>

            <section id="credits" className="scroll-mt-8">
              <h2 className="text-2xl font-bold text-material-text-primary mb-6">
                Credits
              </h2>
              <div className="min-h-[400px]"></div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
