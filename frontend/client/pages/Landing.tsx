import logo from "../../assets/logo.png";
import Navigation from "@/components/Navigation";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

export default function Landing() {
  const navigate = useNavigate();

  const handleExploreClick = () => {
    navigate("/search");
  };

  const handleLearnAboutDataClick = () => {
    navigate("/about#data");
  };

  const handleSeeExamplesClick = () => {
    window.open(
      "https://github.com/PlatosTwin/pandects-app/tree/main/examples",
      "_blank",
    );
  };

  return (
    <div className="flex flex-col min-h-screen bg-cream">
      <Navigation />
      <main className="min-h-[80vh] flex items-center justify-center px-4 py-12">
        <div className="hero-card max-w-[800px] w-full bg-white rounded-3xl shadow-lg p-10 text-center flex flex-col items-center space-y-6 animate-fade-in-up">
          <div className="logo-container">
            <img
              src={logo}
              alt="Pandects Logo"
              className="w-32 h-32 mx-auto rounded-xl object-cover shadow-md"
            />
          </div>

          <h1 className="main-heading text-5xl font-extrabold text-material-text-primary leading-tight">
            Pandects
          </h1>

          <p className="subheading text-xl font-medium text-material-text-secondary">
            Welcome to Pandects, the open-source M&A research platform.
          </p>

          <div className="decorative-divider w-24 h-1 bg-material-blue rounded-full"></div>

          <div className="body-copy max-w-md text-base font-normal text-material-text-secondary leading-relaxed">
            <p>
              What's up with the name? We took a page from Emperor Justinian,
              whose 6th‑century compendium—The Pandects—distilled centuries of
              legal wisdom into a single, authoritative digest.
            </p>
          </div>

          <div className="button-group flex flex-col sm:flex-row gap-4 justify-center items-center w-full">
            <Button
              onClick={handleExploreClick}
              className="cta-button bg-material-blue hover:bg-blue-700 text-white px-8 py-3 rounded-full text-base font-medium transition-colors duration-200"
            >
              Explore Agreements
            </Button>

            <Button
              onClick={handleSeeExamplesClick}
              className="examples-button bg-white hover:bg-gray-50 text-material-blue border-2 border-material-blue px-8 py-3 rounded-full text-base font-medium transition-colors duration-200"
            >
              See Examples
            </Button>

            <Button
              onClick={handleLearnAboutDataClick}
              className="data-button bg-white hover:bg-gray-50 text-material-blue border-2 border-material-blue px-8 py-3 rounded-full text-base font-medium transition-colors duration-200"
            >
              Learn About the Data
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}
