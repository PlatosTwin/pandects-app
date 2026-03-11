import { renderToString } from "react-dom/server";
import { Route, Routes } from "react-router-dom";
import { StaticRouter } from "react-router-dom/server";
import { AppLayout } from "@/components/AppLayout";
import { AuthProvider } from "@/contexts/AuthContext";
import { PRERENDER_ROUTES } from "@shared/route-manifest.mjs";

import About from "@/pages/About";
import BulkData from "@/pages/BulkData";
import Contribute from "@/pages/Contribute";
import Feedback from "@/pages/Feedback";
import Landing from "@/pages/Landing";
import SourcesMethods from "@/pages/SourcesMethods";
import XmlSchema from "@/pages/XmlSchema";
import Taxonomy from "@/pages/Taxonomy";

const PRERENDER_COMPONENTS: Record<string, JSX.Element> = {
  "/": <Landing />,
  "/about": <About />,
  "/bulk-data": <BulkData />,
  "/contribute": <Contribute />,
  "/feedback": <Feedback />,
  "/sources-methods": <SourcesMethods />,
  "/xml-schema": <XmlSchema />,
  "/taxonomy": <Taxonomy />,
};

export function renderPage(pathname: string): string {
  if (!PRERENDER_ROUTES.some((route) => route.pathname === pathname)) {
    throw new Error(`Unsupported prerender path: ${pathname}`);
  }

  const app = (
    <AuthProvider>
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
    </AuthProvider>
  );

  return renderToString(app);
}
