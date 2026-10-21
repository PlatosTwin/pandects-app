import { Github } from "lucide-react";
import { Link } from "react-router-dom";
import { memo } from "react";
import {
  FOOTER_COLUMNS,
  GITHUB_URL,
  LEGAL_LINKS,
  linkKey,
} from "@/components/navigation-links";
import { NavItemLink } from "@/components/NavLinkParts";

const focusRing =
  "rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";

function FooterComponent() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="mt-auto border-t print:hidden border-border bg-background">
      <div className="app-frame">
        <nav
          aria-label="Footer"
          className="grid grid-cols-2 gap-x-6 gap-y-6 py-6 sm:grid-cols-4"
        >
          {FOOTER_COLUMNS.map((column) => (
            <div key={column.heading}>
              <h2 className="text-xs font-semibold uppercase tracking-wide text-foreground">
                {column.heading}
              </h2>
              <ul role="list" className="mt-3 space-y-2">
                {column.links.map((link) => (
                  <li key={linkKey(link)}>
                    <NavItemLink
                      link={link}
                      className={`inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground ${focusRing}`}
                    />
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        <div className="flex flex-col gap-3 border-t border-border py-4 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            {/* The prerender bakes in the build year; after New Year the browser
                renders the next one. Patch the text instead of failing hydration. */}
            <span suppressHydrationWarning>
              © {currentYear} Nikita Bogdanov
            </span>
            <ul
              role="list"
              aria-label="Legal"
              className="flex flex-wrap items-center gap-x-4 gap-y-1"
            >
              {LEGAL_LINKS.map((link) => (
                <li key={link.to}>
                  <Link
                    to={link.to}
                    className={`transition-colors hover:text-foreground ${focusRing}`}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className={`inline-flex self-start text-muted-foreground transition-colors hover:text-foreground sm:self-auto ${focusRing}`}
            title="View source code on GitHub"
            aria-label="View source code on GitHub"
          >
            <Github className="h-4 w-4" aria-hidden="true" />
          </a>
        </div>
      </div>
    </footer>
  );
}

export default memo(FooterComponent);
