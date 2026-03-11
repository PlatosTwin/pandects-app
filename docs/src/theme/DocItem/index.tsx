import React, { Suspense } from "react";

import ExecutionEnvironment from "@docusaurus/ExecutionEnvironment";
import { DocProvider } from "@docusaurus/plugin-content-docs/client";
import { HtmlClassNameProvider } from "@docusaurus/theme-common";
import type { Props } from "@theme/DocItem";
import DocItemLayout from "@theme/DocItem/Layout";
import DocItemMetadata from "@theme/DocItem/Metadata";
import SkeletonLoader from "@theme/SkeletonLoader";
import OriginalDocItem from "@theme-original/DocItem";

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

const LazyApiItem = React.lazy(() => import("@theme/ApiItem"));

function ApiItemFallback(props: Props): JSX.Element {
  const docHtmlClassName = `docs-doc-id-${props.content.metadata.id}`;

  return (
    <DocProvider content={props.content}>
      <HtmlClassNameProvider className={docHtmlClassName}>
        <DocItemMetadata />
        <DocItemLayout>
          <div className="row theme-api-markdown">
            <div className="col col--7 openapi-left-panel__container">
              <SkeletonLoader size="md" />
            </div>
            <div className="col col--5 openapi-right-panel__container">
              <SkeletonLoader size="lg" />
            </div>
          </div>
        </DocItemLayout>
      </HtmlClassNameProvider>
    </DocProvider>
  );
}

export default function DocItem(props: Props): JSX.Element {
  const frontMatter = props.content.frontMatter as Record<string, unknown>;

  if (!isApiDoc(frontMatter)) {
    return <OriginalDocItem {...props} />;
  }

  if (!ExecutionEnvironment.canUseDOM) {
    // SSR/SSG needs the full API renderer synchronously.
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const ApiItem = require("@theme/ApiItem").default;
    return <ApiItem {...props} />;
  }

  return (
    <Suspense fallback={<ApiItemFallback {...props} />}>
      <LazyApiItem {...props} />
    </Suspense>
  );
}
