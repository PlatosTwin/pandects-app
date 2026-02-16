import { renderToString } from "react-dom/server";
import { Route, Routes } from "react-router-dom";
import { StaticRouter } from "react-router-dom/server";
import { AppLayout } from "@/components/AppLayout";
import { AuthProvider } from "@/contexts/AuthContext";

import About from "@/pages/About";
import BulkData from "@/pages/BulkData";
import Contribute from "@/pages/Contribute";
import Feedback from "@/pages/Feedback";
import Landing from "@/pages/Landing";
import SourcesMethods from "@/pages/SourcesMethods";
import XmlSchema from "@/pages/XmlSchema";
import Taxonomy from "@/pages/Taxonomy";

export type PrerenderPath =
  | "/"
  | "/about"
  | "/bulk-data"
  | "/contribute"
  | "/feedback"
  | "/sources-methods"
  | "/xml-schema"
  | "/taxonomy";

export function renderPage(pathname: PrerenderPath): string {
  const app = (
    <AuthProvider>
      <StaticRouter location={pathname}>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Landing />} />
            <Route path="/about" element={<About />} />
            <Route path="/bulk-data" element={<BulkData />} />
            <Route path="/contribute" element={<Contribute />} />
            <Route path="/feedback" element={<Feedback />} />
            <Route path="/sources-methods" element={<SourcesMethods />} />
            <Route path="/xml-schema" element={<XmlSchema />} />
            <Route path="/taxonomy" element={<Taxonomy />} />
          </Route>
        </Routes>
      </StaticRouter>
    </AuthProvider>
  );

  return renderToString(app);
}
