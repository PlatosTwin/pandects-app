import { lazy, Suspense, useMemo } from "react";
import { API_BASE_URL } from "@/lib/api-config";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

// Lazy load SwaggerUI to defer its parsing and rendering
const LazySwaggerUI = lazy(() => import("swagger-ui-react"));

// Import styles at module level (not lazy)
import "swagger-ui-react/swagger-ui.css";
import "./docs.css";

export default function Docs() {
  // Memoize the requestInterceptor to prevent recreating on each render
  const requestInterceptor = useMemo(
    () => (req: any) => {
      const u = new URL(req.url, window.location.origin);
      if (u.pathname.startsWith("/v1/")) {
        req.url = `${API_BASE_URL}${u.pathname}${u.search}`;
      }
      return req;
    },
    []
  );

  return (
    <PageShell size="lg" title="API Docs">
      <Card className="min-h-[640px] p-4 sm:p-6">
        <Suspense
          fallback={
            <div className="space-y-4">
              <Skeleton className="h-8 w-48" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-64 w-full" />
              <Skeleton className="h-32 w-full" />
            </div>
          }
        >
          <LazySwaggerUI
            url="/openapi.yaml"
            requestInterceptor={requestInterceptor}
            deepLinking={false}
            defaultModelsExpandDepth={0}
            tryItOutEnabled={false}
            docExpansion="list"
          />
        </Suspense>
      </Card>
    </PageShell>
  );
}
