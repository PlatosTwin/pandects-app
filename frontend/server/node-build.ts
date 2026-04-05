import { fileURLToPath } from "url";
import { createServer } from "./index";

export function createProductionServer() {
  return createServer();
}

export function startProductionServer() {
  const app = createProductionServer();
  const port = process.env.PORT || 3000;

  const server = app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
  });

  const shutdown = (signal: string) => {
    console.log(`Received ${signal}, shutting down gracefully`);
    server.close(() => process.exit(0));
  };

  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGINT", () => shutdown("SIGINT"));

  return server;
}

const isEntrypoint =
  process.argv[1] != null && fileURLToPath(import.meta.url) === process.argv[1];

if (isEntrypoint) {
  startProductionServer();
}
