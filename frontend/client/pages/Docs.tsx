import Navigation from "@/components/Navigation";
import SwaggerUI from 'swagger-ui-react';
import 'swagger-ui-react/swagger-ui.css';
import { API_BASE_URL } from "@/lib/api-config";

export default function Docs() {
  return (
    <div className="min-h-screen bg-cream flex flex-col">
      <Navigation />
      <main className="flex-1 p-12">
        <div className="max-w-4xl mx-auto">
          <SwaggerUI url="/openapi.yaml" 
          requestInterceptor={(req) => {
        const u = new URL(req.url, window.location.origin);
        if (u.pathname.startsWith("/api/")) {
          req.url = `${API_BASE_URL}${u.pathname}${u.search}`;
        }
        return req;
      }}
          />
        </div>
      </main>
    </div>
  );
}
