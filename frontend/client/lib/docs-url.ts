import brandLinks from "@branding/links.json";

/**
 * Base URL for the docs site: local Docusaurus dev server in development,
 * the production docs site otherwise.
 */
export function getDocsUrl(): string {
  return import.meta.env.DEV ? "http://localhost:3001" : brandLinks.docsSiteUrl;
}
