import { Github } from "lucide-react";
import { useLocation } from "react-router-dom";

export default function Footer() {
  const currentYear = new Date().getFullYear();
  const location = useLocation();
  const isSearchPage = location.pathname === "/";

  return (
    <footer className="mt-auto border-t border-gray-100">
      <div
        className="py-3"
        style={{
          paddingLeft: isSearchPage ? "60px" : "48px",
          paddingRight: "48px",
        }}
      >
        <div className="flex items-center justify-between">
          {/* Copyright on the left */}
          <div className="text-xs text-gray-400">
            © {currentYear} Nikita Bogdanov • Subject to{" "}
            <a
              href="https://www.gnu.org/licenses/gpl-3.0.en.html"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-gray-600 underline"
            >
              GNU GPLv3
            </a>{" "}
            license
          </div>

          {/* GitHub logo in the center - just icon */}
          <div className="flex-1 flex justify-center">
            <a
              href="https://github.com/PlatosTwin/pandects-app"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-gray-600 transition-colors duration-200"
              title="View source code on GitHub"
            >
              <Github className="w-4 h-4" />
            </a>
          </div>

          {/* Empty div for balance */}
          <div className="w-24" />
        </div>
      </div>
    </footer>
  );
}
