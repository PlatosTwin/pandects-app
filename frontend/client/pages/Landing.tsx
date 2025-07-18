import logo from "../../assets/logo.png";
import Navigation from "@/components/Navigation";

export default function Landing() {
  return (
    <div className="h-screen flex flex-col">
      <Navigation />
      <main className="flex-1 flex items-center justify-center px-8">
        <div className="max-w-2xl mx-auto text-center">
          <div className="mb-8">
            <img
              src={logo}
              alt="Pandects Logo"
              className="w-24 h-24 mx-auto rounded-xl object-cover border-2 border-stone-700 shadow-lg"
            />
          </div>

          <h1 className="text-4xl font-bold text-material-text-primary mb-6">
            Pandects
          </h1>

          <div className="text-lg text-material-text-secondary leading-relaxed space-y-4">
            <p>Welcome to Pandects, the open-source M&A research platform.</p>
            <p>
              What's up with the name? We took a page from Emperor Justinian,
              whose 6th‑century compendium—The Pandects—distilled centuries of
              legal wisdom into a single, authoritative digest.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
