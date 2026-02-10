import React from "react";

import { useDoc } from "@docusaurus/plugin-content-docs/client";
import { usePrismTheme } from "@docusaurus/theme-common";
import Translate, { translate } from "@docusaurus/Translate";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import ApiCodeBlock from "@theme/ApiExplorer/ApiCodeBlock";
import { useTypedDispatch, useTypedSelector } from "@theme/ApiItem/hooks";
import SchemaTabs from "@theme/SchemaTabs";
import TabItem from "@theme/TabItem";
import { OPENAPI_RESPONSE } from "@theme/translationIds";
import clsx from "clsx";
import type { ApiItem } from "docusaurus-plugin-openapi-docs/src/types";
import type { ThemeConfig } from "docusaurus-theme-openapi-docs/src/types";

import {
  clearResponse,
  clearCode,
  clearHeaders,
} from "@theme/ApiExplorer/Response/slice";

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

function Response({ item }: { item: ApiItem }) {
  const metadata = useDoc();
  const { siteConfig } = useDocusaurusContext();
  const themeConfig = siteConfig.themeConfig as ThemeConfig;
  const hideSendButton = metadata.frontMatter.hide_send_button;
  const proxy = metadata.frontMatter.proxy ?? themeConfig.api?.proxy;
  const prismTheme = usePrismTheme();
  const code = useTypedSelector((state: any) => state.response.code);
  const headers = useTypedSelector((state: any) => state.response.headers);
  const response = useTypedSelector((state: any) => state.response.value);
  const dispatch = useTypedDispatch();
  const responseStatusClass =
    code &&
    "openapi-response__dot " +
      (parseInt(code) >= 400
        ? "openapi-response__dot--danger"
        : parseInt(code) >= 200 && parseInt(code) < 300
          ? "openapi-response__dot--success"
          : "openapi-response__dot--info");

  if ((!item.servers && !proxy) || hideSendButton) {
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

  return (
    <div className="openapi-explorer__response-container">
      <div className="openapi-explorer__response-title-container">
        <span className="openapi-explorer__response-title">
          {translate({ id: OPENAPI_RESPONSE.TITLE, message: "Response" })}
        </span>
        <span
          className="openapi-explorer__response-clear-btn"
          onClick={() => {
            dispatch(clearResponse());
            dispatch(clearCode());
            dispatch(clearHeaders());
          }}
        >
          {translate({ id: OPENAPI_RESPONSE.CLEAR, message: "Clear" })}
        </span>
      </div>
      <div
        style={{
          backgroundColor:
            (code || hasResponseBody) && prettyResponse !== "Fetching..."
              ? prismTheme.plain.backgroundColor
              : "transparent",
          paddingLeft: "1rem",
          paddingTop: "1rem",
          ...((prettyResponse === "Fetching..." || !code) && {
            paddingBottom: "1rem",
          }),
        }}
      >
        {code && prettyResponse !== "Fetching..." ? (
          <SchemaTabs lazy>
            {/* @ts-ignore */}
            <TabItem
              label={` ${code}`}
              value="body"
              attributes={{
                className: clsx("openapi-response__dot", responseStatusClass),
              }}
              default
            >
              {/* @ts-ignore */}
              <ApiCodeBlock
                className="openapi-explorer__code-block openapi-response__status-code"
                language={response.startsWith("<") ? `xml` : `json`}
              >
                {prettyResponse || ""}
              </ApiCodeBlock>
            </TabItem>
            {/* @ts-ignore */}
            <TabItem
              label={translate({
                id: OPENAPI_RESPONSE.HEADERS_TAB,
                message: "Headers",
              })}
              value="headers"
            >
              {/* @ts-ignore */}
              <ApiCodeBlock
                className="openapi-explorer__code-block openapi-response__status-headers"
                language={response.startsWith("<") ? `xml` : `json`}
              >
                {JSON.stringify(headers, undefined, 2)}
              </ApiCodeBlock>
            </TabItem>
          </SchemaTabs>
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
          <ApiCodeBlock
            className="openapi-explorer__code-block openapi-response__status-code"
            language="text"
          >
            {prettyResponse}
          </ApiCodeBlock>
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
    </div>
  );
}

export default Response;
