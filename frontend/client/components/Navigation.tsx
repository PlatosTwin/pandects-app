import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { isLocalEnvironment } from "@/lib/environment";
import { ChevronDown, Menu } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import logo from "../../assets/logo.png";
import { Button } from "@/components/ui/button";
import PandaEasterEgg from "@/components/PandaEasterEgg";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
  SheetClose,
} from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function Navigation() {
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const navRef = useRef<HTMLElement | null>(null);

  const isActive = (path: string) => location.pathname === path;
  const navLinkBase =
    "rounded-md px-3 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";

  const navLinks = useMemo(
    () => [
      { to: "/search", label: "Search", pandaTarget: "nav-search" },
      { to: "/docs", label: "Docs", pandaTarget: "nav-docs" },
      { to: "/bulk-data", label: "Bulk Data", pandaTarget: "nav-bulk-data" },
      { to: "/about", label: "About", pandaTarget: "nav-about" },
      { to: "/feedback", label: "Feedback" },
    ],
    [],
  );

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/60 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <nav
        ref={navRef}
        className="relative z-0 mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8"
      >
        {/* Brand */}
        <Link
          to="/"
          className="flex items-center gap-3 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
        >
          <img
            src={logo}
            alt="Pandects Logo"
            data-panda-target="logo"
            width={36}
            height={36}
            decoding="async"
            className="relative z-10 h-9 w-9 rounded-md object-cover ring-1 ring-border/60"
          />
          <span
            data-panda-target="brand"
            className="hidden text-base font-semibold tracking-tight text-foreground sm:block"
          >
            Pandects
          </span>
        </Link>

        {/* Desktop navigation */}
        <div className="hidden items-center gap-1 md:flex">
          {navLinks.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              data-panda-target={link.pandaTarget}
              aria-current={isActive(link.to) ? "page" : undefined}
              className={cn(
                navLinkBase,
                isActive(link.to)
                  ? "bg-accent text-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              {link.label}
            </Link>
          ))}

          {isLocalEnvironment() && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  className={cn(
                    "ml-1 h-9 gap-1 px-3 text-sm font-medium",
                    isActive("/editor") && "bg-accent text-foreground",
                  )}
                >
                  Utils
                  <ChevronDown className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuItem asChild>
                  <Link to="/editor">LLM Output Editor</Link>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>

        {/* Mobile navigation */}
        <div className="flex items-center md:hidden">
          <Sheet open={isMobileMenuOpen} onOpenChange={setIsMobileMenuOpen}>
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-10 w-10"
                aria-label="Open menu"
              >
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-[320px] p-0">
              <div className="flex h-full flex-col">
                <SheetHeader className="border-b p-4 text-left">
                  <SheetTitle className="flex items-center gap-3">
                    <img
                      src={logo}
                      alt="Pandects Logo"
                      width={36}
                      height={36}
                      decoding="async"
                      className="h-9 w-9 rounded-md object-cover ring-1 ring-border/60"
                    />
                    <span className="text-base font-semibold tracking-tight">
                      Pandects
                    </span>
                  </SheetTitle>
                </SheetHeader>

                <div className="flex-1 overflow-auto p-2">
                  <div className="grid gap-1">
                    {navLinks.map((link) => (
                      <SheetClose asChild key={link.to}>
                        <Link
                          to={link.to}
                          aria-current={isActive(link.to) ? "page" : undefined}
                          className={cn(
                            navLinkBase,
                            isActive(link.to)
                              ? "bg-accent text-foreground"
                              : "text-muted-foreground hover:bg-accent hover:text-foreground",
                          )}
                        >
                          {link.label}
                        </Link>
                      </SheetClose>
                    ))}
                  </div>

                  {isLocalEnvironment() && (
                    <div className="mt-4 border-t pt-4">
                      <div className="px-3 pb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Utils
                      </div>
                      <div className="grid gap-1">
                        <SheetClose asChild>
                          <Link
                            to="/editor"
                            aria-current={
                              isActive("/editor") ? "page" : undefined
                            }
                            className={cn(
                              navLinkBase,
                              isActive("/editor")
                                ? "bg-accent text-foreground"
                                : "text-muted-foreground hover:bg-accent hover:text-foreground",
                            )}
                          >
                            LLM Output Editor
                          </Link>
                        </SheetClose>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>

        <PandaEasterEgg containerRef={navRef} />
      </nav>
    </header>
  );
}
