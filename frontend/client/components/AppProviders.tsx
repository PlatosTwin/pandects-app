import type { ReactNode } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { AuthProvider } from "@/contexts/AuthContext";
import { FavoritesProvider } from "@/contexts/FavoritesContext";

/**
 * Everything that wraps the router, shared by the browser entry
 * (client/App.tsx, mounted by client/main.tsx) and the prerenderer (prerender/entry.tsx). The two must
 * render the same tree around the page: when the prerender lacked the toaster,
 * every prerendered page failed hydration and React re-rendered the whole app
 * from scratch (a flash, then a blank screen until the code loaded).
 */
export function AppProviders({
  queryClient,
  children,
}: {
  queryClient: QueryClient;
  children: ReactNode;
}) {
  return (
    <ErrorBoundary scope="app-root">
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <FavoritesProvider>
            <TooltipProvider>
              <Toaster />
              {children}
            </TooltipProvider>
          </FavoritesProvider>
        </AuthProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
