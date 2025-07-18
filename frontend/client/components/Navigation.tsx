import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";
import { useState } from "react";
import logo from "../../assets/logo.png";

export default function Navigation() {
  const location = useLocation();
  const [isUtilsOpen, setIsUtilsOpen] = useState(false);

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="bg-gray-800 text-white">
      <div className="max-w-9xl mx-auto px-7">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3">
            <img
              src={logo}
              alt="Pandects Logo"
              className="w-8 h-8 rounded object-cover"
            />
            <span className="text-lg font-semibold">Pandects</span>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center space-x-8">
            {/* Search */}
            <Link
              to="/search"
              className={cn(
                "px-3 py-2 text-sm font-medium transition-colors",
                isActive("/search")
                  ? "text-white"
                  : "text-gray-300 hover:text-white",
              )}
            >
              Search
            </Link>

            {/* Docs */}
            <Link
              to="/docs"
              className={cn(
                "px-3 py-2 text-sm font-medium transition-colors",
                isActive("/docs")
                  ? "text-white"
                  : "text-gray-300 hover:text-white",
              )}
            >
              Docs
            </Link>

            {/* Bulk Data */}
            <Link
              to="/bulk-data"
              className={cn(
                "px-3 py-2 text-sm font-medium transition-colors",
                isActive("/bulk-data")
                  ? "text-white"
                  : "text-gray-300 hover:text-white",
              )}
            >
              Bulk Data
            </Link>

            {/* About */}
            <Link
              to="/about"
              className={cn(
                "px-3 py-2 text-sm font-medium transition-colors",
                isActive("/about")
                  ? "text-white"
                  : "text-gray-300 hover:text-white",
              )}
            >
              About
            </Link>

            {/* Utils Dropdown */}
            <div className="relative">
              <button
                onMouseEnter={() => setIsUtilsOpen(true)}
                onMouseLeave={() => setIsUtilsOpen(false)}
                className={cn(
                  "flex items-center gap-1 px-3 py-2 text-sm font-medium transition-colors",
                  isActive("/editor")
                    ? "text-white"
                    : "text-gray-300 hover:text-white",
                )}
              >
                Utils
                <ChevronDown className="w-4 h-4" />
              </button>

              {isUtilsOpen && (
                <div
                  className="absolute top-full right-0 mt-1 w-48 bg-white border border-gray-200 rounded-md shadow-lg z-50"
                  onMouseEnter={() => setIsUtilsOpen(true)}
                  onMouseLeave={() => setIsUtilsOpen(false)}
                >
                  <Link
                    to="/editor"
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 first:rounded-t-md last:rounded-b-md"
                  >
                    LLM Output Editor
                  </Link>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
