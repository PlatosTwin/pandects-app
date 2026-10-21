import { BrowserRouter, Route, Routes } from "react-router-dom";
import { useEffect, useState, createElement } from "react";
import {
  installGlobalErrorTracking,
  scheduleWhenBrowserIdle,
} from "@/lib/analytics";
import { AppLayout } from "@/components/AppLayout";
import { AppProviders } from "@/components/AppProviders";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { AccessGate } from "@/components/AccessGate";
import { createQueryClient } from "@/lib/query-client";
import { ACCESS_GATED_PATHS } from "@/lib/lockdown";
import { ROUTES } from "@/lib/routes";

export const App = () => {
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
    <AppProviders queryClient={queryClient}>
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
                    ) : ACCESS_GATED_PATHS.has(route.path) ? (
                      <AccessGate>{element}</AccessGate>
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
    </AppProviders>
  );
};
