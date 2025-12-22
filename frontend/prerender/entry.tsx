import { renderToString } from "react-dom/server";
import { Route, Routes } from "react-router-dom";
import { StaticRouter } from "react-router-dom/server";
import { AppLayout } from "@/components/AppLayout";
import { AuthProvider } from "@/contexts/AuthContext";

import About from "@/pages/About";
import BulkData from "@/pages/BulkData";
import Donate from "@/pages/Donate";
import Feedback from "@/pages/Feedback";
import Landing from "@/pages/Landing";

export type PrerenderPath = "/" | "/about" | "/bulk-data" | "/donate" | "/feedback";

export function renderPage(pathname: PrerenderPath): string {
  const app = (
    <AuthProvider>
      <StaticRouter location={pathname}>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Landing />} />
            <Route path="/about" element={<About />} />
            <Route path="/bulk-data" element={<BulkData />} />
            <Route path="/donate" element={<Donate />} />
            <Route path="/feedback" element={<Feedback />} />
          </Route>
        </Routes>
      </StaticRouter>
    </AuthProvider>
  );

  return renderToString(app);
}
