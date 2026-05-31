// Fails the build if nginx.conf or netlify.toml ship a CSP that is missing any
// inline-script hash recorded in shared/csp-script-hashes.generated.json, or if
// they still allow `'unsafe-inline'` for script-src. This is the safety net
// that catches drift between the source HTML / route manifest and the prod
// CSP headers.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const repoRoot = path.resolve(__dirname, "..");
const manifestPath = path.join(repoRoot, "shared", "csp-script-hashes.generated.json");

const TARGETS = [
  { label: "nginx.conf", filePath: path.join(repoRoot, "nginx.conf") },
  { label: "netlify.toml", filePath: path.join(repoRoot, "netlify.toml") },
];

const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const expectedHashes = manifest.hashes;

const errors = [];

for (const { label, filePath } of TARGETS) {
  if (!fs.existsSync(filePath)) {
    errors.push(`${label}: file not found at ${filePath}`);
    continue;
  }
  const text = fs.readFileSync(filePath, "utf8");

  for (const cspMatch of text.matchAll(/Content-Security-Policy[^\n]*?"([^"]+)"/g)) {
    const policy = cspMatch[1];
    const scriptSrcMatch = policy.match(/script-src([^;]*)/);
    if (!scriptSrcMatch) {
      errors.push(`${label}: CSP is missing script-src directive`);
      continue;
    }
    const scriptSrc = scriptSrcMatch[1];

    if (/\b'unsafe-inline'/.test(scriptSrc)) {
      errors.push(`${label}: script-src still contains 'unsafe-inline'`);
    }

    for (const hash of expectedHashes) {
      if (!scriptSrc.includes(`'${hash}'`)) {
        errors.push(`${label}: script-src missing ${hash}`);
      }
    }
  }
}

if (errors.length > 0) {
  process.stderr.write("CSP validation failed:\n");
  for (const message of errors) {
    process.stderr.write(`  - ${message}\n`);
  }
  process.stderr.write(
    "\nRegenerate hashes with `node scripts/compute-csp-script-hashes.mjs` and update nginx.conf / netlify.toml.\n",
  );
  process.exit(1);
}

process.stdout.write(`CSP validation passed (${expectedHashes.length} script hashes present in ${TARGETS.length} files).\n`);
