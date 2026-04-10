import React from "react";

import Head from "@docusaurus/Head";
import Layout from "@theme/Layout";

const GETTING_STARTED_PATH = "/docs/guides/getting-started";

export default function Home() {
  return (
    <Layout title="Pandects Docs">
      <Head>
        <meta
          httpEquiv="refresh"
          content={`0; url=${GETTING_STARTED_PATH}`}
        />
        <meta name="robots" content="noindex,follow" />
        <meta name="googlebot" content="noindex,follow" />
        <link rel="canonical" href={GETTING_STARTED_PATH} />
      </Head>
      <main className="container margin-vert--xl">
        <p>
          Continue to <a href={GETTING_STARTED_PATH}>Getting Started</a>.
        </p>
      </main>
    </Layout>
  );
}
