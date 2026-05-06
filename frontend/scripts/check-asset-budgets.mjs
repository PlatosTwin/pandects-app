import fs from "node:fs";
import path from "node:path";

const assetsDir = path.resolve("dist/spa/assets");
const failures = [];

const jsBudgetBytes = 425_000;
const cssBudgetBytes = 125_000;

const entryJs = findSingleFile(path.join(assetsDir, "js"), /^index-.*\.js$/);
const entryCss = findSingleFile(path.join(assetsDir, "css"), /^index-.*\.css$/);

checkSize(entryJs, jsBudgetBytes, "entry JavaScript");
checkSize(entryCss, cssBudgetBytes, "entry CSS");

if (failures.length > 0) {
  console.error("Performance budget validation failed:");
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log(
  `Validated asset budgets: ${path.basename(entryJs)} <= ${jsBudgetBytes} bytes, ${path.basename(entryCss)} <= ${cssBudgetBytes} bytes.`,
);

function findSingleFile(dir, pattern) {
  const file = fs.readdirSync(dir).find((entry) => pattern.test(entry));
  if (!file) {
    throw new Error(`Expected a file matching ${pattern} in ${dir}`);
  }
  return path.join(dir, file);
}

function checkSize(filePath, maxBytes, label) {
  const size = fs.statSync(filePath).size;
  if (size > maxBytes) {
    failures.push(`${label} exceeded budget: ${size} bytes > ${maxBytes} bytes (${filePath})`);
  }
}
