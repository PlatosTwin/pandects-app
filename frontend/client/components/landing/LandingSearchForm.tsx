import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { trackEvent } from "@/lib/analytics";

/**
 * URL the landing search submits to. Search reads `q` as the keyword query;
 * no `mode` is set, so Search opens in its default (Sections) mode. An empty
 * or whitespace-only query goes to plain `/search`.
 */
export function buildLandingSearchHref(text: string): string {
  const query = text.trim();
  if (!query) return "/search";
  return `/search?${new URLSearchParams({ q: query }).toString()}`;
}

/**
 * Landing hero search box. It is a real GET form to /search with a `q`
 * field, so it also works before hydration; once React is running, submit
 * goes through the router instead of a full page load.
 */
export function LandingSearchForm() {
  const navigate = useNavigate();
  const [text, setText] = useState("");

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const href = buildLandingSearchHref(text);
    trackEvent("landing_search_submit", {
      has_query: href !== "/search",
      to_path: "/search",
    });
    navigate(href);
  };

  return (
    <form
      role="search"
      action="/search"
      method="get"
      onSubmit={onSubmit}
      className="flex w-full items-stretch gap-2"
    >
      <label htmlFor="landing-search-input" className="sr-only">
        Search
      </label>
      <div className="relative min-w-0 flex-1">
        <Search
          className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden="true"
        />
        <input
          id="landing-search-input"
          name="q"
          type="search"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Search clauses, parties, sections…"
          autoComplete="off"
          enterKeyHint="search"
          className="h-12 w-full min-w-0 rounded-lg border border-input bg-background pl-10 pr-3 text-base text-foreground shadow-raised ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        />
      </div>
      <Button type="submit" className="h-12 shrink-0 px-5 text-base">
        Search
      </Button>
    </form>
  );
}
