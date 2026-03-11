import React from "react";

import { DocProvider } from "@docusaurus/plugin-content-docs/client";
import { HtmlClassNameProvider } from "@docusaurus/theme-common";
import ApiDocItemLayout from "@theme/ApiItem/Layout";
import CodeBlock from "@theme/CodeBlock";
import type { Props } from "@theme/DocItem";
import DocItemMetadata from "@theme/DocItem/Metadata";
import SkeletonLoader from "@theme/SkeletonLoader";
import OriginalDocItem from "@theme-original/DocItem";
import clsx from "clsx";
import { Provider } from "react-redux";

function isApiDoc(frontMatter: Record<string, unknown> | undefined): boolean {
  if (!frontMatter) {
    return false;
  }

  return Boolean(
    frontMatter.api ||
      frontMatter.schema ||
      frontMatter.sample ||
      frontMatter.info_path
  );
}

function ApiItemFallback(props: Props): JSX.Element {
  const store = React.useMemo(createStaticApiStore, []);
  const docHtmlClassName = `docs-doc-id-${props.content.metadata.id}`;
  const MDXComponent = props.content;
  const frontMatter = MDXComponent.frontMatter as Record<string, unknown> & {
    sample?: unknown;
    schema?: boolean;
    title?: string;
  };
  const isSchemaDoc = Boolean(frontMatter.schema);
  const sample = frontMatter.sample;

  return (
    <DocProvider content={props.content}>
      <HtmlClassNameProvider className={docHtmlClassName}>
        <DocItemMetadata />
        <ApiDocItemLayout>
          <Provider store={store}>
            <div className={clsx("row", "theme-api-markdown")}>
              <div
                className={clsx(
                  "col col--7 openapi-left-panel__container",
                  isSchemaDoc && "schema"
                )}
              >
                <MDXComponent />
              </div>
              <div className="col col--5 openapi-right-panel__container">
                {isSchemaDoc ? (
                  <CodeBlock language="json" title={String(frontMatter.title ?? "")}>
                    {JSON.stringify(sample ?? {}, null, 2)}
                  </CodeBlock>
                ) : (
                  <SkeletonLoader size="lg" />
                )}
              </div>
            </div>
          </Provider>
        </ApiDocItemLayout>
      </HtmlClassNameProvider>
    </DocProvider>
  );
}

function createStaticApiStore() {
  const state = {
    accept: {
      value: undefined,
      options: [],
    },
    contentType: {
      value: undefined,
      options: [],
    },
    response: {
      value: undefined,
      code: undefined,
      headers: undefined,
    },
    server: {
      value: undefined,
      options: [],
    },
    body: {
      type: "empty" as const,
    },
    params: {
      path: [],
      query: [],
      header: [],
      cookie: [],
    },
    auth: {
      data: {},
      options: {},
      selected: undefined,
    },
    schemaSelection: {
      selections: {},
    },
  };

  return {
    getState: () => state,
    subscribe: () => () => undefined,
    dispatch: (action: unknown) => action,
  };
}

function ClientApiItem(props: Props): JSX.Element {
  const [ApiItem, setApiItem] =
    React.useState<React.ComponentType<Props> | null>(null);

  React.useEffect(() => {
    let mounted = true;

    void import("@theme/ApiItem").then((module) => {
      if (mounted) {
        setApiItem(() => module.default);
      }
    });

    return () => {
      mounted = false;
    };
  }, []);

  if (!ApiItem) {
    return <ApiItemFallback {...props} />;
  }

  return <ApiItem {...props} />;
}

export default function DocItem(props: Props): JSX.Element {
  const frontMatter = props.content.frontMatter as Record<string, unknown>;

  if (!isApiDoc(frontMatter)) {
    return <OriginalDocItem {...props} />;
  }

  return <ClientApiItem {...props} />;
}
