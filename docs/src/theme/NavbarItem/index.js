import React from "react";

import ExecutionEnvironment from "@docusaurus/ExecutionEnvironment";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import OriginalNavbarItem from "@theme-original/NavbarItem";

const LOCAL_APP_URL = "http://localhost:8080/sections";

function isLocalHostname(hostname) {
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname.endsWith(".local")
  );
}

export default function NavbarItem(props) {
  const { siteConfig } = useDocusaurusContext();
  const mainSiteUrl =
    siteConfig.customFields?.mainSiteUrl || "https://pandects.org";
  const appHref = `${mainSiteUrl}/sections`;

  let href = props.href;
  if (
    href === appHref &&
    ExecutionEnvironment.canUseDOM &&
    isLocalHostname(window.location.hostname)
  ) {
    href = LOCAL_APP_URL;
  }

  return <OriginalNavbarItem {...props} href={href} />;
}
