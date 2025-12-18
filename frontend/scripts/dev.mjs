import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

function resolveBin(binName) {
  const binPath =
    process.platform === "win32"
      ? path.resolve("node_modules", ".bin", `${binName}.cmd`)
      : path.resolve("node_modules", ".bin", binName);
  if (!fs.existsSync(binPath)) {
    throw new Error(`Missing ${binName} binary at ${binPath}. Did you run \`npm install\`?`);
  }
  return binPath;
}

function ensureLocalhostCert(certDir) {
  fs.mkdirSync(certDir, { recursive: true });
  const certPath = path.join(certDir, "localhost.pem");
  const keyPath = path.join(certDir, "localhost-key.pem");

  if (fs.existsSync(certPath) && fs.existsSync(keyPath)) {
    return { certPath, keyPath };
  }

  const openssl = spawnSync("openssl", ["version"], { encoding: "utf8" });
  if (openssl.status !== 0) {
    throw new Error(
      "OpenSSL is required to generate a local HTTPS cert (install it, or provide VITE_DEV_HTTPS_CERT and VITE_DEV_HTTPS_KEY).",
    );
  }

  const args = [
    "req",
    "-x509",
    "-newkey",
    "rsa:2048",
    "-sha256",
    "-days",
    "3650",
    "-nodes",
    "-keyout",
    keyPath,
    "-out",
    certPath,
    "-subj",
    "/CN=localhost",
    "-addext",
    "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1",
  ];

  const gen = spawnSync("openssl", args, { stdio: "inherit" });
  if (gen.status !== 0) {
    throw new Error("Failed to generate local HTTPS certificate via OpenSSL.");
  }

  return { certPath, keyPath };
}

const certDir = path.resolve(".cert");
const { certPath, keyPath } = ensureLocalhostCert(certDir);
const vite = resolveBin("vite");

const env = {
  ...process.env,
  VITE_DEV_HTTPS_CERT: process.env.VITE_DEV_HTTPS_CERT || certPath,
  VITE_DEV_HTTPS_KEY: process.env.VITE_DEV_HTTPS_KEY || keyPath,
  VITE_API_BASE_URL: process.env.VITE_API_BASE_URL || "/api",
  VITE_DEV_HOST: process.env.VITE_DEV_HOST || "localhost",
};

const result = spawnSync(vite, process.argv.slice(2), { stdio: "inherit", env });
process.exit(result.status ?? 1);

