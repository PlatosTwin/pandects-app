/* ============================================================================
 * Copyright (c) Palo Alto Networks
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * ========================================================================== */

import React, { Suspense, useEffect, useMemo, useRef } from "react";

import BrowserOnly from "@docusaurus/BrowserOnly";
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

import { createStoreWithState, createStoreWithoutState } from "./store";

interface ApiFrontMatter extends DocFrontMatter {
  readonly api?: string;
}

interface SchemaFrontMatter extends DocFrontMatter {
  readonly schema?: boolean;
}

interface SampleFrontMatter extends DocFrontMatter {
  readonly sample?: any;
}

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

const LazyApiExplorerPanel = React.lazy(() => import("./ApiExplorerPanel"));

function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return bytes;
}

export default function ApiItem(props: Props): JSX.Element {
  const docHtmlClassName = `docs-doc-id-${props.content.metadata.id}`;
  const MDXComponent = props.content;
  const { frontMatter } = MDXComponent;
  const { info_path: infoPath } = frontMatter as DocFrontMatter;
  const { api } = frontMatter as ApiFrontMatter;
  const { schema } = frontMatter as SchemaFrontMatter;
  const { sample } = frontMatter as SampleFrontMatter;
  const rightPanelRef = useRef<HTMLDivElement | null>(null);
  const { siteConfig } = useDocusaurusContext();
  const themeConfig = siteConfig.themeConfig as ThemeConfig;
  const options = themeConfig.api;
  const isBrowser = useIsBrowser();

  const parsedApi = useMemo(() => {
    if (!api) {
      return undefined;
    }

    const decodedApi = JSON.parse(
      new TextDecoder().decode(ungzip(base64ToUint8Array(api)))
    ) as ApiItemType;

    if (!Array.isArray(decodedApi.servers) || decodedApi.servers.length === 0) {
      decodedApi.servers = DEFAULT_API_SERVERS;
    }

    return decodedApi;
  }, [api]);

  const store = useMemo(() => {
    const persistenceMiddleware = createPersistenceMiddleware(options);

    if (!isBrowser || !parsedApi) {
      return createStoreWithoutState({}, [persistenceMiddleware]);
    }

    const acceptOptions = Object.entries(parsedApi.responses ?? {})
      .filter(([statusCode]) => /2\d\d/.test(statusCode))
      .flatMap(([, response]) => Object.keys(response.content ?? {}));
    const contentTypeOptions = Object.keys(parsedApi.requestBody?.content ?? {});
    const params = {
      path: [] as ParameterObject[],
      query: [] as ParameterObject[],
      header: [] as ParameterObject[],
      cookie: [] as ParameterObject[],
    };

    parsedApi.parameters?.forEach((param) => {
      params[param.in].push(param as ParameterObject);
    });

    const auth = createAuth({
      security: parsedApi.security,
      securitySchemes: parsedApi.securitySchemes,
      options,
    });
    const storage = createStorage(options?.authPersistence ?? "sessionStorage");
    const storedServer = storage.getItem("server");
    const serverValue = storedServer
      ? (JSON.parse(storedServer) as ServerObject)
      : undefined;

    return createStoreWithState(
      {
        accept: {
          value: acceptOptions[0],
          options: acceptOptions,
        },
        contentType: {
          value: contentTypeOptions[0],
          options: contentTypeOptions,
        },
        server: {
          value: serverValue?.url ? serverValue : undefined,
          options: parsedApi.servers ?? [],
        },
        response: { value: undefined },
        body: { type: "empty" },
        params,
        auth,
        schemaSelection: { selections: {} },
      },
      [persistenceMiddleware]
    );
  }, [isBrowser, options, parsedApi]);

  useEffect(() => {
    const panel = rightPanelRef.current;
    if (!panel) {
      return;
    }

    let rafId = 0;
    let initialRafId = 0;
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
    initialRafId = window.requestAnimationFrame(() => {
      rafId = window.requestAnimationFrame(() => {
        rafId = 0;
        scheduleUpdate();
      });
    });

    return () => {
      if (initialRafId !== 0) {
        window.cancelAnimationFrame(initialRafId);
      }
      if (rafId !== 0) {
        window.cancelAnimationFrame(rafId);
      }
      resizeObserver.disconnect();
      window.removeEventListener("resize", scheduleUpdate);
    };
  }, [api, schema, sample]);

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
                    {() => (
                      <Suspense fallback={<SkeletonLoader size="lg" />}>
                        <LazyApiExplorerPanel
                          api={parsedApi!}
                          infoPath={infoPath}
                        />
                      </Suspense>
                    )}
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
