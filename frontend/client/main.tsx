import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot, hydrateRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { lazy, useEffect } from "react";
import {
  bootstrapAnalytics,
  installGlobalErrorTracking,
  loadAnalyticsScript,
} from "@/lib/analytics";
import Landing from "./pages/Landing";
import { AppLayout } from "@/components/AppLayout";
import { AuthProvider } from "@/contexts/AuthContext";

const SearchPage = lazy(() => import("./pages/Search"));
const BulkData = lazy(() => import("./pages/BulkData"));
const AgreementIndex = lazy(() => import("./pages/AgreementIndex"));
const About = lazy(() => import("./pages/About"));
const SourcesMethods = lazy(() => import("./pages/SourcesMethods"));
const XmlSchema = lazy(() => import("./pages/XmlSchema"));
const Taxonomy = lazy(() => import("./pages/Taxonomy"));
const Leaderboards = lazy(() => import("./pages/Leaderboards"));
const Feedback = lazy(() => import("./pages/Feedback"));
const Contribute = lazy(() => import("./pages/Contribute"));
const NotFound = lazy(() => import("./pages/NotFound"));
const Account = lazy(() => import("./pages/Account"));
const AuthGoogleCallback = lazy(() => import("./pages/AuthGoogleCallback"));
const ForgotPassword = lazy(() => import("./pages/ForgotPassword"));
const PrivacyPolicy = lazy(() => import("./pages/PrivacyPolicy"));
const ResetPassword = lazy(() => import("./pages/ResetPassword"));
const VerifyEmail = lazy(() => import("./pages/VerifyEmail"));
const Terms = lazy(() => import("./pages/Terms"));
const License = lazy(() => import("./pages/License"));
const SoftwareLicense = lazy(() => import("./pages/SoftwareLicense"));
const DataLicense = lazy(() => import("./pages/DataLicense"));

bootstrapAnalytics();

const App = () => {
  useEffect(() => {
    const schedule = window.requestIdleCallback
      ? window.requestIdleCallback(() => loadAnalyticsScript())
      : window.setTimeout(() => loadAnalyticsScript(), 1500);
    return () => {
      if (window.cancelIdleCallback) {
        window.cancelIdleCallback(schedule as number);
      } else {
        window.clearTimeout(schedule as number);
      }
    };
  }, []);

  useEffect(() => installGlobalErrorTracking(), []);

  return (
    <AuthProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/" element={<Landing />} />
              <Route path="/search" element={<SearchPage />} />
              <Route path="/bulk-data" element={<BulkData />} />
              <Route path="/agreement-index" element={<AgreementIndex />} />
              <Route path="/sources-methods" element={<SourcesMethods />} />
              <Route path="/xml-schema" element={<XmlSchema />} />
              <Route path="/taxonomy" element={<Taxonomy />} />
              <Route path="/leaderboards" element={<Leaderboards />} />
              <Route path="/about" element={<About />} />
              <Route path="/feedback" element={<Feedback />} />
              <Route path="/contribute" element={<Contribute />} />
              <Route path="/account" element={<Account />} />
              <Route path="/auth/forgot-password" element={<ForgotPassword />} />
                <Route path="/privacy-policy" element={<PrivacyPolicy />} />
                <Route path="/auth/reset-password" element={<ResetPassword />} />
                <Route path="/auth/verify-email" element={<VerifyEmail />} />
                <Route path="/terms" element={<Terms />} />
                <Route path="/license" element={<License />} />
                <Route path="/license/software" element={<SoftwareLicense />} />
                <Route path="/license/data" element={<DataLicense />} />
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
  );
};

const container = document.getElementById("root")!;
if (container.hasChildNodes()) {
  hydrateRoot(container, <App />);
} else {
  createRoot(container).render(<App />);
}
