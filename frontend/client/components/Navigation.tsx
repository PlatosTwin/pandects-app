import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { FileText, Search } from "lucide-react";
import logo from "../../assets/logo.png";

export default function Navigation() {
  const location = useLocation();

  const navItems = [
    {
      path: "/",
      label: "M&A Clause Search",
      icon: Search,
      description: "Search M&A agreement clauses",
    },
    {
      path: "/editor",
      label: "LLM Output Editor",
      icon: FileText,
      description: "Edit and manage LLM outputs",
    },
  ];

  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm">
      <div className="px-12 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img
              src={logo}
              alt="Pandects Logo"
              className="w-10 h-10 rounded-lg object-cover border-2 border-stone-700"
            />
            <div className="flex flex-col">
              <span className="text-lg font-semibold text-material-text-primary">
                Pandects
              </span>
              <span className="text-xs text-material-text-secondary">
                M&A Research Platform
              </span>
            </div>
          </div>

          {/* Navigation Items */}
          <div className="flex items-center space-x-8">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;

              return (
                <Link
                  key={item.path}
                  to={item.path}
                  tabIndex={item.path === "/" ? 7 : 8}
                  className={cn(
                    "flex items-center gap-3 px-4 py-2 rounded-md transition-all duration-200",
                    isActive
                      ? "bg-material-blue text-white shadow-md focus:outline-none focus:ring-2 focus:ring-material-blue focus:ring-offset-2 focus:ring-offset-cream"
                      : "text-material-text-primary hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-material-blue focus:ring-offset-2",
                  )}
                >
                  <Icon className="w-5 h-5" />
                  <div className="flex flex-col">
                    <span className="text-sm font-medium">{item.label}</span>
                    <span
                      className={cn(
                        "text-xs",
                        isActive
                          ? "text-blue-100"
                          : "text-material-text-secondary",
                      )}
                    >
                      {item.description}
                    </span>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
