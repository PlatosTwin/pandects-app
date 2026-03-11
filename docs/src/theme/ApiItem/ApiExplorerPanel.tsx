import React from "react";

import ApiExplorer from "@theme/ApiExplorer";
import type { ApiItem as ApiItemType } from "docusaurus-plugin-openapi-docs/src/types";

export default function ApiExplorerPanel({
  api,
  infoPath,
}: {
  api: ApiItemType;
  infoPath?: string;
}): JSX.Element {
  return <ApiExplorer item={api} infoPath={infoPath} />;
}
