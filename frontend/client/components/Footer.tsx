import { Github } from "lucide-react";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="mt-auto border-t border-border/60 bg-background/70 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
        <div className="text-xs text-muted-foreground">
          © {currentYear} Nikita Bogdanov •{" "}
          <a
            href="https://www.gnu.org/licenses/gpl-3.0.en.html"
            target="_blank"
            rel="noopener noreferrer"
            className="underline underline-offset-2 hover:text-foreground"
          >
            GNU GPLv3
          </a>
        </div>

        <a
          href="https://github.com/PlatosTwin/pandects-app"
          target="_blank"
          rel="noopener noreferrer"
          className="text-muted-foreground hover:text-foreground transition-colors"
          title="View source code on GitHub"
          aria-label="View source code on GitHub"
        >
          <Github className="h-4 w-4" />
        </a>
      </div>
    </footer>
  );
}
