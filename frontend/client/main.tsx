import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot, hydrateRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { lazy, useEffect } from "react";
import { installGlobalErrorTracking } from "@/lib/analytics";
import { apiUrl } from "@/lib/api-config";
import Search from "./pages/Search";
import Landing from "./pages/Landing";
import { AppLayout } from "@/components/AppLayout";
import { AuthProvider } from "@/contexts/AuthContext";

const Docs = lazy(() => import("./pages/Docs"));
const BulkData = lazy(() => import("./pages/BulkData"));
const AgreementIndex = lazy(() => import("./pages/AgreementIndex"));
const About = lazy(() => import("./pages/About"));
const SourcesMethods = lazy(() => import("./pages/SourcesMethods"));
const Feedback = lazy(() => import("./pages/Feedback"));
const Donate = lazy(() => import("./pages/Donate"));
const NotFound = lazy(() => import("./pages/NotFound"));
const Account = lazy(() => import("./pages/Account"));
const AuthGoogleCallback = lazy(() => import("./pages/AuthGoogleCallback"));
const ForgotPassword = lazy(() => import("./pages/ForgotPassword"));
const PrivacyPolicy = lazy(() => import("./pages/PrivacyPolicy"));
const ResetPassword = lazy(() => import("./pages/ResetPassword"));
const Terms = lazy(() => import("./pages/Terms"));
const License = lazy(() => import("./pages/License"));

const queryClient = new QueryClient();

const App = () => {
  // on first mount, fire a lightweight ping to warm up the Fly.io db machine
  useEffect(() => {
    void fetch(apiUrl("api/dumps")).catch(() => undefined);
  }, []);

  useEffect(() => installGlobalErrorTracking(), []);

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              <Route element={<AppLayout />}>
                <Route path="/" element={<Landing />} />
                <Route path="/search" element={<Search />} />
                <Route path="/docs" element={<Docs />} />
                <Route path="/bulk-data" element={<BulkData />} />
                <Route path="/agreement-index" element={<AgreementIndex />} />
                <Route path="/sources-methods" element={<SourcesMethods />} />
                <Route path="/about" element={<About />} />
                <Route path="/feedback" element={<Feedback />} />
                <Route path="/donate" element={<Donate />} />
                <Route path="/account" element={<Account />} />
                <Route path="/auth/forgot-password" element={<ForgotPassword />} />
                <Route path="/privacy-policy" element={<PrivacyPolicy />} />
                <Route path="/auth/reset-password" element={<ResetPassword />} />
                <Route path="/terms" element={<Terms />} />
                <Route path="/license" element={<License />} />
                <Route
                  path="/auth/google/callback"
                  element={<AuthGoogleCallback />}
                />
                {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                <Route path="*" element={<NotFound />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
};

const container = document.getElementById("root")!;
if (container.hasChildNodes()) {
  hydrateRoot(container, <App />);
} else {
  createRoot(container).render(<App />);
}
