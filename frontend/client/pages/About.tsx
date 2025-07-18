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
      const scrollPosition = window.scrollY + 150; // Adjust offset for better detection

      // Check if we're at the bottom of the page
      if (
        window.innerHeight + window.scrollY >=
        document.body.offsetHeight - 100
      ) {
        setActiveSection("credits");
        return;
      }

      // Find the current section
      let currentSection = "overview"; // Default fallback

      for (let i = sections.length - 1; i >= 0; i--) {
        const sectionId = sections[i];
        const element = document.getElementById(sectionId);
        if (element) {
          const offsetTop = element.offsetTop;

          if (scrollPosition >= offsetTop) {
            currentSection = sectionId;
            break;
          }
        }
      }

      setActiveSection(currentSection);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="min-h-screen bg-cream">
      <Navigation />

      <div className="flex">
        {/* Fixed Navigation Pane */}
        <nav className="fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 overflow-y-auto z-10">
          <div className="p-6">
            <ul className="space-y-2">
              <li>
                <button
                  onClick={() => scrollToSection("overview")}
                  className={cn(
                    "w-full text-left px-3 py-2 text-sm rounded-md transition-colors",
                    activeSection === "overview"
                      ? "bg-blue-50 text-blue-700 font-medium"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50",
                  )}
                >
                  Overview
                </button>
              </li>
              <li>
                <button
                  onClick={() => scrollToSection("data")}
                  className={cn(
                    "w-full text-left px-3 py-2 text-sm rounded-md transition-colors",
                    activeSection === "data"
                      ? "bg-blue-50 text-blue-700 font-medium"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50",
                  )}
                >
                  Data
                </button>
              </li>
              <li className="ml-4">
                <button
                  onClick={() => scrollToSection("sources")}
                  className={cn(
                    "w-full text-left px-3 py-2 text-sm rounded-md transition-colors",
                    activeSection === "sources"
                      ? "bg-blue-50 text-blue-700 font-medium"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50",
                  )}
                >
                  Sources
                </button>
              </li>
              <li className="ml-4">
                <button
                  onClick={() => scrollToSection("processing-pipelines")}
                  className={cn(
                    "w-full text-left px-3 py-2 text-sm rounded-md transition-colors",
                    activeSection === "processing-pipelines"
                      ? "bg-blue-50 text-blue-700 font-medium"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50",
                  )}
                >
                  Processing pipelines
                </button>
              </li>
              <li>
                <button
                  onClick={() => scrollToSection("contributing")}
                  className={cn(
                    "w-full text-left px-3 py-2 text-sm rounded-md transition-colors",
                    activeSection === "contributing"
                      ? "bg-blue-50 text-blue-700 font-medium"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50",
                  )}
                >
                  Contributing
                </button>
              </li>
              <li>
                <button
                  onClick={() => scrollToSection("credits")}
                  className={cn(
                    "w-full text-left px-3 py-2 text-sm rounded-md transition-colors",
                    activeSection === "credits"
                      ? "bg-blue-50 text-blue-700 font-medium"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50",
                  )}
                >
                  Credits
                </button>
              </li>
            </ul>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 ml-64">
          <div className="max-w-4xl mx-auto px-8 py-8">
            <div className="space-y-16">
              {/* Overview Section */}
              <section id="overview" className="scroll-mt-8">
                <h1 className="text-3xl font-bold text-gray-900 mb-6">
                  Overview
                </h1>
                <div className="min-h-[400px]"></div>
              </section>

              {/* Data Section */}
              <section id="data" className="scroll-mt-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Data</h2>
                <div className="min-h-[300px]"></div>

                {/* Sources Subsection */}
                <div id="sources" className="scroll-mt-8 mt-12">
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">
                    Sources
                  </h3>
                  <div className="min-h-[300px]"></div>
                </div>

                {/* Processing Pipelines Subsection */}
                <div id="processing-pipelines" className="scroll-mt-8 mt-12">
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">
                    Processing pipelines
                  </h3>
                  <div className="min-h-[300px]"></div>
                </div>
              </section>

              {/* Contributing Section */}
              <section id="contributing" className="scroll-mt-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">
                  Contributing
                </h2>
                <div className="min-h-[400px]"></div>
              </section>

              {/* Credits Section */}
              <section id="credits" className="scroll-mt-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">
                  Credits
                </h2>
                <div className="min-h-[400px]"></div>
              </section>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
