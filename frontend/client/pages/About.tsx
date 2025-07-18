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
      const scrollPosition = window.scrollY + 150;

      if (
        window.innerHeight + window.scrollY >=
        document.body.offsetHeight - 100
      ) {
        setActiveSection("credits");
        return;
      }

      let currentSection = "overview";
      for (let i = sections.length - 1; i >= 0; i--) {
        const el = document.getElementById(sections[i]);
        if (el && scrollPosition >= el.offsetTop) {
          currentSection = sections[i];
          break;
        }
      }

      setActiveSection(currentSection);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (sectionId: string) => {
    const el = document.getElementById(sectionId);
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-cream">
      {/* --- Header --- */}
      <Navigation />

      <div className="flex">
        {/* --- Sidebar (only on md+) --- */}
        <nav
          className="
            hidden md:block
            fixed top-16 bottom-16 left-0
            w-64
            bg-white
            border-r border-gray-200
            overflow-y-auto
            z-10
          "
        >
          <div className="p-6">
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
                        ? "bg-blue-50 text-blue-700 font-medium"
                        : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
                    )}
                  >
                    {label}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </nav>

        {/* --- Main content --- */}
        <main className="flex-1 ml-0 md:ml-64">
          <div className="max-w-4xl mx-auto px-8 py-8 space-y-16">
            <section id="overview" className="scroll-mt-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-6">
                Overview
              </h1>
              {/* just empty space now—no box styling */}
              <div className="min-h-[400px]"></div>
            </section>

            <section id="data" className="scroll-mt-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Data</h2>
              <div className="min-h-[300px]"></div>

              <div id="sources" className="scroll-mt-8 mt-12">
                <h3 className="text-xl font-semibold text-gray-900 mb-4">
                  Sources
                </h3>
                <div className="min-h-[300px]"></div>
              </div>

              <div id="processing-pipelines" className="scroll-mt-8 mt-12">
                <h3 className="text-xl font-semibold text-gray-900 mb-4">
                  Processing pipelines
                </h3>
                <div className="min-h-[300px]"></div>
              </div>
            </section>

            <section id="contributing" className="scroll-mt-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">
                Contributing
              </h2>
              <div className="min-h-[400px]"></div>
            </section>

            <section id="credits" className="scroll-mt-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">
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
