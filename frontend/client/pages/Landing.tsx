import logo from "../../assets/logo.png";
import Navigation from "@/components/Navigation";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

export default function Landing() {
  const navigate = useNavigate();

  const handleExploreClick = () => {
    navigate("/search");
  };

  return (
    <div className="flex flex-col h-full">
      <Navigation />
      <main
        className="min-h-[80vh] flex items-center justify-center px-4 py-12"
        style={{ backgroundColor: "#F5F7FA" }}
      >
        <div className="hero-card max-w-[800px] w-full bg-white rounded-3xl shadow-lg p-10 text-center flex flex-col items-center space-y-6 animate-fade-in-up">
          <div className="logo-container">
            <img
              src={logo}
              alt="Pandects Logo"
              className="w-32 h-32 mx-auto rounded-xl object-cover shadow-md"
            />
          </div>

          <h1 className="main-heading text-5xl font-extrabold text-black leading-tight">
            Pandects
          </h1>

          <p className="subheading text-xl font-medium text-gray-600">
            Welcome to Pandects, the open-source M&A research platform.
          </p>

          <div className="decorative-divider w-24 h-1 bg-material-blue rounded-full"></div>

          <div className="body-copy max-w-md text-base font-normal text-gray-500 leading-relaxed">
            <p>
              What's up with the name? We took a page from Emperor Justinian,
              whose 6th‑century compendium—The Pandects—distilled centuries of
              legal wisdom into a single, authoritative digest.
            </p>
          </div>

          <Button
            onClick={handleExploreClick}
            className="cta-button bg-material-blue hover:bg-blue-700 text-white px-8 py-3 rounded-full text-base font-medium transition-colors duration-200"
          >
            Explore Agreements
          </Button>
        </div>
      </main>
    </div>
  );
}
