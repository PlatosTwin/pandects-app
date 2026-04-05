import { readFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import React from 'react';
import { renderAsync } from '@react-email/render';

type TemplateComponent = (props: Record<string, unknown>) => JSX.Element & {
  PreviewProps?: Record<string, unknown>;
};

async function main(): Promise<void> {
  const templateId = process.argv[2];
  if (!templateId) {
    throw new Error('Missing template id.');
  }

  const raw = readFileSync(0, 'utf8');
  const props = raw.trim() ? (JSON.parse(raw) as Record<string, unknown>) : {};
  const templatesDir = path.resolve(process.cwd(), 'emails');
  const templatePath = path.join(templatesDir, `${templateId}.tsx`);
  const templateModule = await import(pathToFileURL(templatePath).href);
  const templateName = templateId
    .split(/[^a-zA-Z0-9]/)
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join('');
  const Template = (templateModule.default ?? templateModule[templateName]) as
    | TemplateComponent
    | undefined;

  if (!Template) {
    throw new Error(`Template ${templateId} does not have a default export.`);
  }

  const html = await renderAsync(React.createElement(Template, props));
  process.stdout.write(html);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
});
