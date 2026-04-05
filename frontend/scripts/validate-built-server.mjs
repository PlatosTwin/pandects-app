import path from "node:path";
import { pathToFileURL } from "node:url";

const modulePath = path.resolve("dist/server/node-build.mjs");
const builtServer = await import(pathToFileURL(modulePath).href);

if (typeof builtServer.createProductionServer !== "function") {
  throw new Error("Built server bundle is missing createProductionServer().");
}

const app = builtServer.createProductionServer();

if (typeof app?.handle !== "function" || typeof app?.use !== "function") {
  throw new Error("Built server bundle did not return a valid Express app.");
}

console.log("Validated built production server bundle.");
