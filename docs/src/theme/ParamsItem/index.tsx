import React from "react";

import { translate } from "@docusaurus/Translate";
import { Example } from "@theme/Example";
import Markdown from "@theme/Markdown";
import { OPENAPI_SCHEMA_ITEM } from "@theme/translationIds";
import clsx from "clsx";

import {
  getQualifierMessage,
  getSchemaName,
} from "docusaurus-theme-openapi-docs/lib/markdown/schema";
import { guard } from "docusaurus-theme-openapi-docs/lib/markdown/utils";

export interface ExampleObject {
  summary?: string;
  description?: string;
  value?: any;
  externalValue?: string;
}

export interface Props {
  className: string;
  param: {
    description?: string;
    example: any;
    examples: Record<string, ExampleObject> | undefined;
    name: string;
    required: boolean;
    deprecated: boolean;
    schema: any;
    enumDescriptions?: [string, string][];
  };
}

const PENDING_PREFIX = /^Pending:\s*/i;

function normalizePendingDescription(description: string): string {
  const withoutPrefix = description.replace(PENDING_PREFIX, "");
  return withoutPrefix.replace(/^\s*([a-z])/, (match, firstLetter: string) =>
    match.replace(firstLetter, firstLetter.toUpperCase())
  );
}

const getEnumDescriptionMarkdown = (enumDescriptions?: [string, string][]) => {
  if (enumDescriptions?.length) {
    const enumValue = translate({
      id: OPENAPI_SCHEMA_ITEM.ENUM_VALUE,
      message: "Enum Value",
    });
    const description = translate({
      id: OPENAPI_SCHEMA_ITEM.ENUM_DESCRIPTION,
      message: "Description",
    });
    return `| ${enumValue} | ${description} |
| ---- | ----- |
${enumDescriptions
  .map((desc) => {
    return `| ${desc[0]} | ${desc[1]} | `.replaceAll("\n", "<br/>");
  })
  .join("\n")}
    `;
  }

  return "";
};

function ParamsItem({ param }: Props) {
  const { name, required, deprecated, enumDescriptions } = param;
  const description = param.description ?? "";
  const isPending = PENDING_PREFIX.test(description);
  const displayDescription = isPending
    ? normalizePendingDescription(description)
    : description;

  let schema = param.schema;
  let defaultValue: string | undefined;

  const examples = param.examples ?? (schema?.examples as any[] | undefined);
  const example = param.example ?? schema?.example;

  if (!schema) {
    schema = { type: "any" };
  }

  if (!schema.type) {
    schema.type = "any";
  }

  if (schema) {
    if (schema.items) {
      defaultValue = schema.items.default;
    } else {
      defaultValue = schema.default;
    }
  }

  const renderSchemaName = guard(schema, (value) => (
    <span className="openapi-schema__type"> {getSchemaName(value)}</span>
  ));

  const renderSchemaRequired = guard(required, () => (
    <span className="openapi-schema__required">
      {translate({ id: OPENAPI_SCHEMA_ITEM.REQUIRED, message: "required" })}
    </span>
  ));

  const renderDeprecated = guard(deprecated, () => (
    <span className="openapi-schema__deprecated">
      {translate({ id: OPENAPI_SCHEMA_ITEM.DEPRECATED, message: "deprecated" })}
    </span>
  ));

  const renderPending = guard(isPending, () => (
    <span className="openapi-schema__pending">Pending</span>
  ));

  const renderQualifier = guard(getQualifierMessage(schema), (qualifier) => (
    <Markdown>{qualifier}</Markdown>
  ));

  const renderDescription = guard(displayDescription, (value) => (
    <Markdown>{value}</Markdown>
  ));

  const renderEnumDescriptions = guard(
    getEnumDescriptionMarkdown(enumDescriptions),
    (value) => {
      return (
        <div style={{ marginTop: ".5rem" }}>
          <Markdown>{value}</Markdown>
        </div>
      );
    }
  );

  function renderDefaultValue() {
    if (defaultValue !== undefined) {
      if (typeof defaultValue === "string") {
        return (
          <div>
            <strong>
              {translate({
                id: OPENAPI_SCHEMA_ITEM.DEFAULT_VALUE,
                message: "Default value:",
              })}{" "}
            </strong>
            <span>
              <code>{defaultValue}</code>
            </span>
          </div>
        );
      }
      return (
        <div>
          <strong>
            {translate({
              id: OPENAPI_SCHEMA_ITEM.DEFAULT_VALUE,
              message: "Default value:",
            })}{" "}
          </strong>
          <span>
            <code>{JSON.stringify(defaultValue)}</code>
          </span>
        </div>
      );
    }
    return undefined;
  }

  return (
    <div className="openapi-params__list-item">
      <span className="openapi-schema__container">
        <strong
          className={clsx("openapi-schema__property", {
            "openapi-schema__strikethrough": deprecated,
          })}
        >
          {name}
        </strong>
        {renderSchemaName}
        {(required || deprecated || isPending) && (
          <span className="openapi-schema__divider"></span>
        )}
        {renderSchemaRequired}
        {renderPending}
        {renderDeprecated}
      </span>
      {renderQualifier}
      {renderDescription}
      {renderEnumDescriptions}
      {renderDefaultValue()}
      <Example example={example} />
      <Example examples={examples} />
    </div>
  );
}

export default ParamsItem;
