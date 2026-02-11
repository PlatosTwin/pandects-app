import React, { type ReactNode } from "react";
import { translate } from "@docusaurus/Translate";
import type { Props } from "@theme/DocRoot/Layout/Sidebar/ExpandButton";

import styles from "./styles.module.css";

function ExpandChevron({ className }: { className?: string }): ReactNode {
  return (
    <svg
      viewBox="0 0 20 20"
      width="20"
      height="20"
      aria-hidden="true"
      focusable="false"
      className={className}
    >
      <path
        d="M5.5 4.5L10.5 10L5.5 15.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M10.5 4.5L15.5 10L10.5 15.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function DocRootLayoutSidebarExpandButton({
  toggleSidebar,
}: Props): ReactNode {
  const expandSidebarLabel = translate({
    id: "theme.docs.sidebar.expandButtonAriaLabel",
    message: "Expand sidebar",
    description:
      "The ARIA label and title attribute for expand button of doc sidebar",
  });

  return (
    <button
      type="button"
      className={styles.expandButton}
      title={translate({
        id: "theme.docs.sidebar.expandButtonTitle",
        message: "Expand sidebar",
        description:
          "The ARIA label and title attribute for expand button of doc sidebar",
      })}
      aria-label={expandSidebarLabel}
      onClick={toggleSidebar}
    >
      <ExpandChevron className={styles.expandButtonIcon} />
    </button>
  );
}
