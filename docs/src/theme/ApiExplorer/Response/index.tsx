import React from "react";

import { useDoc } from "@docusaurus/plugin-content-docs/client";
import Translate, { translate } from "@docusaurus/Translate";
import ApiCodeBlock from "@theme/ApiExplorer/ApiCodeBlock";
import { useTypedDispatch, useTypedSelector } from "@theme/ApiItem/hooks";
import { OPENAPI_RESPONSE } from "@theme/translationIds";
import clsx from "clsx";
import type { ApiItem } from "docusaurus-plugin-openapi-docs/src/types";

import {
  clearResponse,
  clearCode,
  clearHeaders,
} from "@theme/ApiExplorer/Response/slice";

const ApiCodeBlockComponent = ApiCodeBlock as React.ComponentType<any>;

function formatXml(xml: string) {
  const tab = "  ";
  let formatted = "";
  let indent = "";

  xml.split(/>\s*</).forEach((node) => {
    if (node.match(/^\/\w/)) {
      indent = indent.substring(tab.length);
    }
    formatted += indent + "<" + node + ">\r\n";
    if (node.match(/^<?\w[^>]*[^/]$/)) {
      indent += tab;
    }
  });
  return formatted.substring(1, formatted.length - 3);
}

function statusLabel(code: string): string {
  const numeric = Number.parseInt(code, 10);
  if (Number.isNaN(numeric)) {
    return "Response";
  }
  if (numeric >= 500) {
    return "Server Error";
  }
  if (numeric >= 400) {
    return "Client Error";
  }
  if (numeric >= 300) {
    return "Redirect";
  }
  if (numeric >= 200) {
    return "Success";
  }
  return "Info";
}

function Response({ item }: { item: ApiItem }) {
  const metadata = useDoc();
  const hideSendButton = metadata.frontMatter.hide_send_button;
  const code = useTypedSelector((state: any) => state.response.code);
  const response = useTypedSelector((state: any) => state.response.value);
  const dispatch = useTypedDispatch();
  const responseStatusClass =
    code &&
    (parseInt(code) >= 400
      ? "openapi-response__dot--danger"
      : parseInt(code) >= 200 && parseInt(code) < 300
        ? "openapi-response__dot--success"
        : "openapi-response__dot--info");

  if (hideSendButton) {
    return null;
  }

  let prettyResponse: string = response;

  if (prettyResponse) {
    try {
      prettyResponse = JSON.stringify(JSON.parse(response), null, 2);
    } catch {
      if (response.startsWith("<")) {
        prettyResponse = formatXml(response);
      }
    }
  }

  const hasResponseBody =
    Boolean(prettyResponse) && prettyResponse !== "Fetching...";
  const hasStatusCode = Boolean(code) && prettyResponse !== "Fetching...";

  return (
    <div className="openapi-explorer__response-container">
      <div className="openapi-explorer__response-title-container">
        <span className="openapi-explorer__response-title">
          {translate({ id: OPENAPI_RESPONSE.TITLE, message: "Response" })}
        </span>
        {hasStatusCode && (
          <div className="openapi-explorer__response-title-meta">
            <span
              className={clsx(
                "openapi-explorer__response-status",
                "openapi-response__dot",
                responseStatusClass
              )}
            >
              {`${code} - ${statusLabel(code)}`}
            </span>
            <span className="openapi-explorer__response-example-chip">
              Example
            </span>
          </div>
        )}
      </div>
      <div className="openapi-explorer__response-body">
        {code && prettyResponse !== "Fetching..." ? (
          <ApiCodeBlockComponent
            className="openapi-explorer__code-block openapi-response__status-code"
            language={response.startsWith("<") ? `xml` : `json`}
            showLineNumbers
          >
            {prettyResponse || ""}
          </ApiCodeBlockComponent>
        ) : prettyResponse === "Fetching..." ? (
          <div className="openapi-explorer__loading-container">
            <div className="openapi-response__lds-ring">
              <div></div>
              <div></div>
              <div></div>
              <div></div>
            </div>
          </div>
        ) : hasResponseBody ? (
          // Show network/CORS/validation errors even when there is no HTTP status code.
          // The upstream theme otherwise falls back to the placeholder text and hides the real error.
          <ApiCodeBlockComponent
            className="openapi-explorer__code-block openapi-response__status-code"
            language="text"
            showLineNumbers
          >
            {prettyResponse}
          </ApiCodeBlockComponent>
        ) : (
          <p className="openapi-explorer__response-placeholder-message">
            <Translate
              id={OPENAPI_RESPONSE.PLACEHOLDER}
              values={{ sendApiRequest: <code>Send API Request</code> }}
            >
              {"Click the {sendApiRequest} button above and see the response here!"}
            </Translate>
          </p>
        )}
      </div>
      <button
        type="button"
        className="openapi-explorer__response-clear-btn"
        onClick={() => {
          dispatch(clearResponse());
          dispatch(clearCode());
          dispatch(clearHeaders());
        }}
      >
        {translate({ id: OPENAPI_RESPONSE.CLEAR, message: "Clear" })}
      </button>
    </div>
  );
}

export default Response;
