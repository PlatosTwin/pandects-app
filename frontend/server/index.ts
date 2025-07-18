import express from "express";
import cors from "cors";
import path from "path";
import { handleDemo } from "./routes/demo";

export function createServer() {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Example API routes
  app.get("/api/ping", (_req, res) => {
    res.json({ message: "Hello from Express server v2!" });
  });

  app.get("/api/demo", handleDemo);

  // Serve static files in production
  if (process.env.NODE_ENV === "production") {
    const staticPath = path.join(__dirname, "../spa");
    app.use(express.static(staticPath));

    // SPA fallback - serve index.html for all non-API routes
    app.get("*", (_req, res) => {
      res.sendFile(path.join(staticPath, "index.html"));
    });
  }

  return app;
}
