import SwaggerUI from "swagger-ui-react";
import "swagger-ui-react/swagger-ui.css";
import "./docs.css";
import { API_BASE_URL } from "@/lib/api-config";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";

export default function Docs() {
  return (
    <PageShell size="lg" title="API Docs" subtitle="Explore the Pandects API via OpenAPI.">
      <Card className="min-h-[640px] p-4 sm:p-6">
        <SwaggerUI
          url="/openapi.yaml"
          requestInterceptor={(req) => {
            const u = new URL(req.url, window.location.origin);
            if (u.pathname.startsWith("/api/")) {
              req.url = `${API_BASE_URL}${u.pathname}${u.search}`;
            }
            return req;
          }}
        />
      </Card>
    </PageShell>
  );
}
