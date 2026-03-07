import React from "react";
import { useEffect } from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import OriginalNavbar from "@theme-original/Navbar";

export default function NavbarWrapper(props) {
  const { siteConfig } = useDocusaurusContext();
  useEffect(() => {
    const mainSiteUrl = siteConfig.customFields?.mainSiteUrl || "https://pandects.org";
    const appLink = document.querySelector(`a[href="${mainSiteUrl}/sections"]`);
    if (!appLink) return;

    const hostname = window.location.hostname;
    const isLocal =
      hostname === "localhost" || hostname === "127.0.0.1" || hostname.endsWith(".local");

    appLink.setAttribute("href", isLocal ? "http://localhost:8080/sections" : `${mainSiteUrl}/sections`);
  }, [siteConfig.customFields]);

  return <OriginalNavbar {...props} />;
}
