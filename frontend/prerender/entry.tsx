import { renderToString } from "react-dom/server";
import { Route, Routes } from "react-router-dom";
import { StaticRouter } from "react-router-dom/server";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppLayout } from "@/components/AppLayout";
import { AuthProvider } from "@/contexts/AuthContext";
import { PRERENDER_ROUTES } from "@shared/route-manifest.mjs";

import About from "@/pages/About";
import AgreementIndex from "@/pages/AgreementIndex";
import BulkData from "@/pages/BulkData";
import Contribute from "@/pages/Contribute";
import DataLicense from "@/pages/DataLicense";
import Feedback from "@/pages/Feedback";
import Landing from "@/pages/Landing";
import Leaderboards from "@/pages/Leaderboards";
import License from "@/pages/License";
import PrivacyPolicy from "@/pages/PrivacyPolicy";
import Search from "@/pages/Search";
import SoftwareLicense from "@/pages/SoftwareLicense";
import SourcesMethods from "@/pages/SourcesMethods";
import Taxonomy from "@/pages/Taxonomy";
import Terms from "@/pages/Terms";
import TrendsAnalyses from "@/pages/TrendsAnalyses";
import XmlSchema from "@/pages/XmlSchema";

const PRERENDER_COMPONENTS: Record<string, JSX.Element> = {
  "/": <Landing />,
  "/about": <About />,
  "/agreement-index": <AgreementIndex />,
  "/bulk-data": <BulkData />,
  "/contribute": <Contribute />,
  "/feedback": <Feedback />,
  "/leaderboards": <Leaderboards />,
  "/license": <License />,
  "/license/data": <DataLicense />,
  "/license/software": <SoftwareLicense />,
  "/privacy-policy": <PrivacyPolicy />,
  "/search": <Search />,
  "/sources-methods": <SourcesMethods />,
  "/taxonomy": <Taxonomy />,
  "/terms": <Terms />,
  "/trends-analyses": <TrendsAnalyses />,
  "/xml-schema": <XmlSchema />,
};

export function renderPage(pathname: string): string {
  if (!PRERENDER_ROUTES.some((route) => route.pathname === pathname)) {
    throw new Error(`Unsupported prerender path: ${pathname}`);
  }

  const app = (
    <AuthProvider>
      <TooltipProvider>
        <StaticRouter location={pathname}>
          <Routes>
            <Route element={<AppLayout />}>
              {PRERENDER_ROUTES.map((route) => (
                <Route
                  key={route.pathname}
                  path={route.pathname}
                  element={PRERENDER_COMPONENTS[route.pathname]}
                />
              ))}
            </Route>
          </Routes>
        </StaticRouter>
      </TooltipProvider>
    </AuthProvider>
  );

  return renderToString(app);
}
