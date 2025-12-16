import { renderToString } from "react-dom/server";
import { Route, Routes } from "react-router-dom";
import { StaticRouter } from "react-router-dom/server";
import { AppLayout } from "@/components/AppLayout";

import About from "@/pages/About";
import BulkData from "@/pages/BulkData";

export type PrerenderPath = "/about" | "/bulk-data";

export function renderPage(pathname: PrerenderPath): string {
  const app = (
    <StaticRouter location={pathname}>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/about" element={<About />} />
          <Route path="/bulk-data" element={<BulkData />} />
        </Route>
      </Routes>
    </StaticRouter>
  );

  return renderToString(app);
}

