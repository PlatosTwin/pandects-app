import { Github } from "lucide-react";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-white border-t border-gray-200 mt-auto">
      <div className="px-12 py-6">
        <div className="flex items-center justify-between">
          {/* Copyright on the left */}
          <div className="text-sm text-material-text-secondary">
            © {currentYear} Nikita Bogdanov
          </div>

          {/* GitHub logo in the center */}
          <div className="flex-1 flex justify-center">
            <a
              href="https://github.com/PlatosTwin/pandects-app"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-material-text-secondary hover:text-material-text-primary transition-colors duration-200 group"
              title="View source code on GitHub"
            >
              <Github className="w-5 h-5 group-hover:scale-110 transition-transform duration-200" />
              <span className="text-sm font-medium">GitHub</span>
            </a>
          </div>

          {/* Empty div for balance */}
          <div className="w-32" />
        </div>
      </div>
    </footer>
  );
}
