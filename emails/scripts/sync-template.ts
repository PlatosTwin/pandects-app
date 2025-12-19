import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import dotenv from 'dotenv';
import React from 'react';
import { renderAsync } from '@react-email/render';
import { Resend } from 'resend';

type TemplateComponent = (props: Record<string, unknown>) => JSX.Element & {
  PreviewProps?: Record<string, unknown>;
};

const args = process.argv.slice(2);
const templateId =
  getArgValue(args, '--template-id') ??
  getArgValue(args, '--templateId') ??
  process.env.npm_config_template_id;

dotenv.config({ path: path.resolve(process.cwd(), '.env') });

main(templateId).catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  exitWithError(message);
});

async function main(templateIdValue?: string): Promise<void> {
  if (!templateIdValue) {
    exitWithError(
      'Missing required --template-id. Example: npm run email -- --template-id verify-email',
    );
  }

  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    exitWithError('Missing RESEND_API_KEY in environment.');
  }

  const templatesDir = path.resolve(process.cwd(), 'emails');
  const templatePath = path.join(templatesDir, `${templateIdValue}.tsx`);

  if (!existsSync(templatePath)) {
    exitWithError(`Template file not found: ${templatePath}`);
  }

  const templateModule = await import(pathToFileURL(templatePath).href);
  const Template = (templateModule.default ??
    templateModule[pascalCase(templateIdValue)]) as TemplateComponent | undefined;

  if (!Template) {
    exitWithError(`No default export found in ${templatePath}`);
  }

  const props = Template.PreviewProps ?? {};
  const reactElement = React.createElement(Template, props);
  const resend = new Resend(apiKey);

  const getResponse = await resend.templates.get(templateIdValue);

  if (getResponse.error) {
    if (getResponse.error.statusCode === 404) {
      const html = await renderAsync(reactElement);
      const variables = extractTemplateVariables(html);
      const createResponse = await resend.templates.create({
        name: templateIdValue,
        react: reactElement,
        variables,
      });

      if (createResponse.error) {
        exitWithError(`Create failed: ${createResponse.error.message}`);
      }

      console.log(
        `Created template ${templateIdValue} (${createResponse.data?.id ?? 'unknown id'}).`,
      );
      return;
    }

    exitWithError(`Failed to fetch template: ${getResponse.error.message}`);
  }

  const template = getResponse.data;

  if (!template) {
    exitWithError('Template lookup returned no data.');
  }

  if (template.status !== 'published') {
    exitWithError(
      `Template ${templateIdValue} is ${template.status}; only published templates can be updated.`,
    );
  }

  const html = await renderAsync(reactElement);
  const variables = extractTemplateVariables(html);
  const updateResponse = await resend.templates.update(templateIdValue, { html, variables });

  if (updateResponse.error) {
    exitWithError(`Update failed: ${updateResponse.error.message}`);
  }

  console.log(`Updated template ${templateIdValue}.`);
}

function getArgValue(values: string[], key: string): string | undefined {
  const index = values.indexOf(key);
  if (index === -1) {
    return undefined;
  }
  return values[index + 1];
}

function pascalCase(value: string): string {
  return value
    .split(/[^a-zA-Z0-9]/)
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join('');
}

function exitWithError(message: string): never {
  console.error(message);
  process.exit(1);
}

function extractTemplateVariables(html: string) {
  const pattern = /{{{([A-Z0-9_]+)}}}/g;
  const reserved = new Set([
    'FIRST_NAME',
    'LAST_NAME',
    'EMAIL',
    'RESEND_UNSUBSCRIBE_URL',
    'contact',
    'this',
  ]);
  const found = new Map<string, 'string' | 'number'>();
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(html)) !== null) {
    const key = match[1];
    if (reserved.has(key)) {
      exitWithError(`Template variable ${key} is reserved and cannot be used.`);
    }
    found.set(key, 'string');
  }

  if (found.size === 0) {
    return undefined;
  }

  return Array.from(found.entries()).map(([key, type]) => ({ key, type }));
}
