import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot, hydrateRoot } from "react-dom/client";
import { TooltipProvider } from "@/components/ui/tooltip";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState, createElement } from "react";
import {
  installGlobalErrorTracking,
  scheduleWhenBrowserIdle,
} from "@/lib/analytics";
import { AppLayout } from "@/components/AppLayout";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { AuthProvider } from "@/contexts/AuthContext";
import { FavoritesProvider } from "@/contexts/FavoritesContext";
import { createQueryClient } from "@/lib/query-client";
import { ROUTES } from "@/lib/routes";

const App = () => {
  const [queryClient] = useState(createQueryClient);

  useEffect(() => {
    let cleanup = () => undefined;
    const cancelScheduledInstall = scheduleWhenBrowserIdle(() => {
      cleanup = installGlobalErrorTracking();
    });

    return () => {
      cancelScheduledInstall();
      cleanup();
    };
  }, []);

  return (
    <ErrorBoundary scope="app-root">
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <FavoritesProvider>
            <TooltipProvider>
              <Toaster />
              <BrowserRouter>
                <Routes>
                  <Route element={<AppLayout />}>
                    {ROUTES.map((route) => {
                      const element = createElement(route.component);
                      return (
                        <Route
                          key={route.path}
                          path={route.path}
                          element={
                            route.protected ? (
                              <ProtectedRoute>{element}</ProtectedRoute>
                            ) : (
                              element
                            )
                          }
                        />
                      );
                    })}
                  </Route>
                </Routes>
              </BrowserRouter>
            </TooltipProvider>
          </FavoritesProvider>
        </AuthProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

const container = document.getElementById("root")!;
if (container.hasChildNodes()) {
  hydrateRoot(container, <App />);
} else {
  createRoot(container).render(<App />);
}
