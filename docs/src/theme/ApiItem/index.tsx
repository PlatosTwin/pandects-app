/* ============================================================================
 * Copyright (c) Palo Alto Networks
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * ========================================================================== */

import React, { useEffect, useRef } from "react";

import BrowserOnly from "@docusaurus/BrowserOnly";
import ExecutionEnvironment from "@docusaurus/ExecutionEnvironment";
import { DocProvider } from "@docusaurus/plugin-content-docs/client";
import { HtmlClassNameProvider } from "@docusaurus/theme-common";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import useIsBrowser from "@docusaurus/useIsBrowser";
import { createAuth } from "@theme/ApiExplorer/Authorization/slice";
import { createPersistenceMiddleware } from "@theme/ApiExplorer/persistenceMiddleware";
import { createStorage } from "@theme/ApiExplorer/storage-utils";
import DocItemLayout from "@theme/ApiItem/Layout";
import CodeBlock from "@theme/CodeBlock";
import type { Props } from "@theme/DocItem";
import DocItemMetadata from "@theme/DocItem/Metadata";
import SkeletonLoader from "@theme/SkeletonLoader";
import clsx from "clsx";
import type {
  ParameterObject,
  ServerObject,
} from "docusaurus-plugin-openapi-docs/src/openapi/types";
import type { ApiItem as ApiItemType } from "docusaurus-plugin-openapi-docs/src/types";
import type {
  DocFrontMatter,
  ThemeConfig,
} from "docusaurus-theme-openapi-docs/src/types";
import { ungzip } from "pako";
import { Provider } from "react-redux";

import { createStoreWithoutState, createStoreWithState } from "./store";

const DEFAULT_API_SERVERS = [
  {
    url: "https://api.pandects.org",
    description: "Production API",
  },
  {
    url: "http://localhost:5113",
    description: "Local development API",
  },
];

let ApiExplorer = (_: { item: any; infoPath: any }) => <div />;

if (ExecutionEnvironment.canUseDOM) {
  ApiExplorer = require("@theme/ApiExplorer").default;
}

interface ApiFrontMatter extends DocFrontMatter {
  readonly api?: ApiItemType;
}

interface SchemaFrontMatter extends DocFrontMatter {
  readonly schema?: boolean;
}

interface SampleFrontMatter extends DocFrontMatter {
  readonly sample?: any;
}

function base64ToUint8Array(base64: string) {
  const binary = atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

export default function ApiItem(props: Props): JSX.Element {
  const docHtmlClassName = `docs-doc-id-${props.content.metadata.id}`;
  const MDXComponent = props.content;
  const { frontMatter } = MDXComponent;
  const { info_path: infoPath } = frontMatter as DocFrontMatter;
  let { api } = frontMatter as ApiFrontMatter;
  const { schema } = frontMatter as SchemaFrontMatter;
  const { sample } = frontMatter as SampleFrontMatter;
  if (api) {
    try {
      api = JSON.parse(
        new TextDecoder().decode(ungzip(base64ToUint8Array(api as any)))
      );
      if (!Array.isArray(api.servers) || api.servers.length === 0) {
        api.servers = DEFAULT_API_SERVERS;
      }
    } catch {}
  }
  const { siteConfig } = useDocusaurusContext();
  const themeConfig = siteConfig.themeConfig as ThemeConfig;
  const options = themeConfig.api;
  const isBrowser = useIsBrowser();

  const statusRegex = new RegExp("(20[0-9]|2[1-9][0-9])");

  let store: any = {};
  const persistenceMiddleware = createPersistenceMiddleware(options);
  const rightPanelRef = useRef<HTMLDivElement | null>(null);

  if (!isBrowser) {
    store = createStoreWithoutState({}, [persistenceMiddleware]);
  }

  if (isBrowser) {
    let acceptArray: any = [];
    for (const [code, content] of Object.entries(api?.responses ?? [])) {
      if (statusRegex.test(code)) {
        acceptArray.push(Object.keys(content.content ?? {}));
      }
    }
    acceptArray = acceptArray.flat();

    const content = api?.requestBody?.content ?? {};
    const contentTypeArray = Object.keys(content);
    const servers = api?.servers ?? [];
    const params = {
      path: [] as ParameterObject[],
      query: [] as ParameterObject[],
      header: [] as ParameterObject[],
      cookie: [] as ParameterObject[],
    };
    api?.parameters?.forEach(
      (param: { in: "path" | "query" | "header" | "cookie" }) => {
        const paramType = param.in;
        const paramsArray: ParameterObject[] = params[paramType];
        paramsArray.push(param as ParameterObject);
      }
    );
    const auth = createAuth({
      security: api?.security,
      securitySchemes: api?.securitySchemes,
      options,
    });

    const storage = createStorage(options?.authPersistence ?? "sessionStorage");
    const server = storage.getItem("server");
    const serverObject = server
      ? (JSON.parse(server) as ServerObject)
      : undefined;

    store = createStoreWithState(
      {
        accept: {
          value: acceptArray[0],
          options: acceptArray,
        },
        contentType: {
          value: contentTypeArray[0],
          options: contentTypeArray,
        },
        server: {
          value: serverObject?.url ? serverObject : undefined,
          options: servers,
        },
        response: { value: undefined },
        body: { type: "empty" },
        params,
        auth,
        schemaSelection: { selections: {} },
      },
      [persistenceMiddleware]
    );
  }

  useEffect(() => {
    if (!isBrowser) {
      return;
    }
    const panel = rightPanelRef.current;
    if (!panel) {
      return;
    }

    let rafId = 0;
    const scheduleUpdate = () => {
      if (rafId !== 0) {
        return;
      }
      rafId = window.requestAnimationFrame(() => {
        rafId = 0;
        if (window.innerWidth <= 996) {
          panel.style.removeProperty("--openapi-right-panel-top");
          return;
        }

        const rootStyle = window.getComputedStyle(document.documentElement);
        const rootFontPx = Number.parseFloat(rootStyle.fontSize) || 16;
        const navbarHeightPx =
          Number.parseFloat(rootStyle.getPropertyValue("--ifm-navbar-height")) || 0;
        const viewportGapPx = 0.95 * rootFontPx;
        const minTopPx = navbarHeightPx + viewportGapPx;

        const panelHeightPx = panel.getBoundingClientRect().height;
        const desiredTopPx = window.innerHeight - panelHeightPx - viewportGapPx;
        const stickyTopPx = Math.min(minTopPx, desiredTopPx);

        panel.style.setProperty("--openapi-right-panel-top", `${stickyTopPx}px`);
      });
    };

    const resizeObserver = new ResizeObserver(scheduleUpdate);
    resizeObserver.observe(panel);
    window.addEventListener("resize", scheduleUpdate, { passive: true });
    scheduleUpdate();

    return () => {
      if (rafId !== 0) {
        window.cancelAnimationFrame(rafId);
      }
      resizeObserver.disconnect();
      window.removeEventListener("resize", scheduleUpdate);
    };
  }, [isBrowser]);

  if (api) {
    return (
      <DocProvider content={props.content}>
        <HtmlClassNameProvider className={docHtmlClassName}>
          <DocItemMetadata />
          <DocItemLayout>
            <Provider store={store}>
              <div className={clsx("row", "theme-api-markdown")}>
                <div className="col col--7 openapi-left-panel__container">
                  <MDXComponent />
                </div>
                <div
                  ref={rightPanelRef}
                  className="col col--5 openapi-right-panel__container"
                >
                  <BrowserOnly fallback={<SkeletonLoader size="lg" />}>
                    {() => {
                      return <ApiExplorer item={api} infoPath={infoPath} />;
                    }}
                  </BrowserOnly>
                </div>
              </div>
            </Provider>
          </DocItemLayout>
        </HtmlClassNameProvider>
      </DocProvider>
    );
  } else if (schema) {
    return (
      <DocProvider content={props.content}>
        <HtmlClassNameProvider className={docHtmlClassName}>
          <DocItemMetadata />
          <DocItemLayout>
            <div className={clsx("row", "theme-api-markdown")}>
              <div className="col col--7 openapi-left-panel__container schema">
                <MDXComponent />
              </div>
              <div
                ref={rightPanelRef}
                className="col col--5 openapi-right-panel__container"
              >
                <CodeBlock language="json" title={`${frontMatter.title}`}>
                  {JSON.stringify(sample, null, 2)}
                </CodeBlock>
              </div>
            </div>
          </DocItemLayout>
        </HtmlClassNameProvider>
      </DocProvider>
    );
  }

  // Non-API docs
  return (
    <DocProvider content={props.content}>
      <HtmlClassNameProvider className={docHtmlClassName}>
        <DocItemMetadata />
        <DocItemLayout>
          <div className="row">
            <div className="col col--12 markdown">
              <MDXComponent />
            </div>
          </div>
        </DocItemLayout>
      </HtmlClassNameProvider>
    </DocProvider>
  );
}
