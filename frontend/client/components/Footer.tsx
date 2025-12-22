import { Github } from "lucide-react";
import { Link } from "react-router-dom";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="mt-auto border-t border-border/60 bg-background/70 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex flex-col items-center justify-between gap-3 sm:flex-row">
          <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-center text-xs leading-relaxed text-muted-foreground sm:justify-start sm:text-left">
            <span>© {currentYear} Nikita Bogdanov</span>
            <span aria-hidden="true" className="opacity-60">
              •
            </span>
            <Link
              to="/license"
              className="underline underline-offset-4 decoration-muted-foreground/50 hover:text-foreground hover:decoration-foreground/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-sm"
            >
              License
            </Link>
            <span aria-hidden="true" className="opacity-60">
              •
            </span>
            <nav aria-label="Legal" className="flex items-center gap-x-3">
              <Link
                to="/privacy-policy"
                className="underline underline-offset-4 decoration-muted-foreground/50 hover:text-foreground hover:decoration-foreground/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-sm"
              >
                Privacy Policy
              </Link>
              <span aria-hidden="true" className="opacity-60">
                •
              </span>
              <Link
                to="/terms"
                className="underline underline-offset-4 decoration-muted-foreground/50 hover:text-foreground hover:decoration-foreground/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-sm"
              >
                Terms
              </Link>
            </nav>
          </div>

          <a
            href="https://github.com/PlatosTwin/pandects-app"
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-sm"
            title="View source code on GitHub"
            aria-label="View source code on GitHub"
          >
            <Github className="h-4 w-4" />
          </a>
        </div>
      </div>
    </footer>
  );
}
