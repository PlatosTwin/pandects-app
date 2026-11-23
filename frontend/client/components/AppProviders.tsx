import {
  lazy,
  Suspense,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import type { QueryClient } from "@tanstack/react-query";
import { QueryClientProvider } from "@tanstack/react-query";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { AuthProvider } from "@/contexts/AuthContext";
import { FavoritesProvider } from "@/contexts/FavoritesContext";

/**
 * Everything that wraps the router, shared by the browser entry
 * (client/App.tsx, mounted by client/main.tsx) and the prerenderer (prerender/entry.tsx). The two must
 * render the same tree around the page: when the prerender lacked the toaster,
 * every prerendered page failed hydration and React re-rendered the whole app
 * from scratch (a flash, then a blank screen until the code loaded).
 *
 * The toaster mounts only after hydration (see DeferredToaster), so the
 * prerendered HTML and the first client render both have no toaster.
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
            <DeferredToaster />
            {children}
          </FavoritesProvider>
        </AuthProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

const subscribeNever = () => () => {};

const Toaster = lazy(() =>
  import("@/components/ui/toaster").then((module) => ({
    default: module.Toaster,
  })),
);

/**
 * Loads the toaster (Radix Toast) after the first render instead of in the
 * entry bundle. Toasts raised before it loads wait in the use-toast store and
 * show once it mounts.
 */
function DeferredToaster() {
  // false on the server and during hydration, true once the client runs.
  const mounted = useSyncExternalStore(
    subscribeNever,
    () => true,
    () => false,
  );
  if (!mounted) return null;
  return (
    <Suspense fallback={null}>
      <Toaster />
    </Suspense>
  );
}
