import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useEffect } from "react";
import { isLocalEnvironment } from "./lib/environment";
import Edit from "./pages/Edit";
import Search from "./pages/Search";
import Landing from "./pages/Landing";
import Docs from "./pages/Docs";
import BulkData from "./pages/BulkData";
import About from "./pages/About";
import Feedback from "./pages/Feedback";
import NotFound from "./pages/NotFound";
import Footer from "./components/Footer";
import SiteBanner from "./components/SiteBanner";

const queryClient = new QueryClient();

const App = () => {
  // on first mount, fire a lightweight ping to warm up the Fly.io db machine
  useEffect(() => {
    fetch("/api/dumps").catch(() => {
    });
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <div className="min-h-screen flex flex-col">
            <SiteBanner />
            <main className="flex-1">
              <Routes>
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
              </Routes>
            </main>
            <Footer />
          </div>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

createRoot(document.getElementById("root")!).render(<App />);
