import React, { type ReactNode } from "react";
import clsx from "clsx";
import { translate } from "@docusaurus/Translate";
import type { Props } from "@theme/DocSidebar/Desktop/CollapseButton";

import styles from "./styles.module.css";

function CollapseChevron({ className }: { className?: string }): ReactNode {
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

export default function CollapseButton({ onClick }: Props): ReactNode {
  return (
    <button
      type="button"
      title={translate({
        id: "theme.docs.sidebar.collapseButtonTitle",
        message: "Collapse sidebar",
        description: "The title attribute for collapse button of doc sidebar",
      })}
      aria-label={translate({
        id: "theme.docs.sidebar.collapseButtonAriaLabel",
        message: "Collapse sidebar",
        description: "The title attribute for collapse button of doc sidebar",
      })}
      className={clsx(
        "button button--secondary button--outline",
        styles.collapseSidebarButton
      )}
      onClick={onClick}
    >
      <CollapseChevron className={styles.collapseSidebarButtonIcon} />
    </button>
  );
}
