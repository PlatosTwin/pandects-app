import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { lazy, useEffect } from "react";
import { isLocalEnvironment } from "./lib/environment";
import Search from "./pages/Search";
import Landing from "./pages/Landing";
import { AppLayout } from "@/components/AppLayout";

const Docs = lazy(() => import("./pages/Docs"));
const BulkData = lazy(() => import("./pages/BulkData"));
const About = lazy(() => import("./pages/About"));
const Feedback = lazy(() => import("./pages/Feedback"));
const NotFound = lazy(() => import("./pages/NotFound"));
const Edit = lazy(() => import("./pages/Edit"));

const queryClient = new QueryClient();

const App = () => {
  // on first mount, fire a lightweight ping to warm up the Fly.io db machine
  useEffect(() => {
    void fetch("/api/dumps").catch(() => undefined);
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
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
              <Route path="/about" element={<About />} />
              <Route path="/feedback" element={<Feedback />} />
              {/* Editor route - Only available in local development */}
              {isLocalEnvironment() && (
                <Route path="/editor" element={<Edit />} />
              )}
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

createRoot(document.getElementById("root")!).render(<App />);
