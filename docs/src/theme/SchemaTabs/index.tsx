/* ============================================================================
 * Copyright (c) Palo Alto Networks
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * ========================================================================== */

import React, {
  cloneElement,
  ReactElement,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  sanitizeTabsChildren,
  TabProps,
  TabsProvider,
  useScrollPositionBlocker,
  useTabs,
  useTabsContextValue,
} from "@docusaurus/theme-common/internal";
import { TabItemProps } from "@docusaurus/theme-common/lib/utils/tabsUtils";
import useIsBrowser from "@docusaurus/useIsBrowser";
import clsx from "clsx";
import flatten from "lodash/flatten";

export interface SchemaTabsProps extends TabProps {
  onChange?: (index: number) => void;
}

function TabList({
  className,
  block,
  selectedValue,
  selectValue,
  tabValues,
  onChange,
}: SchemaTabsProps & ReturnType<typeof useTabs>) {
  const tabRefs = useRef<(HTMLLIElement | null)[]>([]);
  const tabItemListContainerRef = useRef<HTMLUListElement>(null);
  const [showTabArrows, setShowTabArrows] = useState(false);
  const { blockElementScrollPositionUntilNextRender } =
    useScrollPositionBlocker();

  const handleTabChange = (
    event:
      | React.FocusEvent<HTMLLIElement>
      | React.MouseEvent<HTMLLIElement>
      | React.KeyboardEvent<HTMLLIElement>
  ) => {
    const newTab = event.currentTarget;
    const newTabIndex = tabRefs.current.indexOf(newTab);
    const newTabValue = tabValues[newTabIndex]!.value;

    if (newTabValue !== selectedValue) {
      blockElementScrollPositionUntilNextRender(newTab);
      selectValue(newTabValue);
      onChange?.(newTabIndex);
    }
  };

  const handleKeydown = (event: React.KeyboardEvent<HTMLLIElement>) => {
    let focusElement: HTMLLIElement | null = null;

    switch (event.key) {
      case "Enter":
        handleTabChange(event);
        break;
      case "ArrowRight": {
        const nextTab = tabRefs.current.indexOf(event.currentTarget) + 1;
        focusElement = tabRefs.current[nextTab] ?? tabRefs.current[0]!;
        break;
      }
      case "ArrowLeft": {
        const prevTab = tabRefs.current.indexOf(event.currentTarget) - 1;
        focusElement =
          tabRefs.current[prevTab] ??
          tabRefs.current[tabRefs.current.length - 1]!;
        break;
      }
      default:
        break;
    }

    focusElement?.focus();
  };

  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        requestAnimationFrame(() => {
          setShowTabArrows(entry.target.clientWidth < entry.target.scrollWidth);
        });
      }
    });

    if (!tabItemListContainerRef.current) {
      return undefined;
    }

    resizeObserver.observe(tabItemListContainerRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <div
      className="openapi-tabs__schema-tabs-container"
      style={{ marginBottom: "1rem" }}
    >
      {showTabArrows && (
        <button
          type="button"
          className="openapi-tabs__arrow left"
          aria-label="Scroll schema tabs left"
          onClick={(event) => {
            event.preventDefault();
            tabItemListContainerRef.current!.scrollLeft -= 90;
          }}
        />
      )}
      <ul
        ref={tabItemListContainerRef}
        role="tablist"
        aria-orientation="horizontal"
        className={clsx(
          "openapi-tabs__schema-list-container",
          "tabs",
          {
            "tabs--block": block,
          },
          className
        )}
      >
        {tabValues.map(({ value, label, attributes }) => (
          <li
            role="tab"
            tabIndex={selectedValue === value ? 0 : -1}
            aria-selected={selectedValue === value}
            key={value}
            ref={(tabControl) => {
              tabRefs.current[
                tabValues.findIndex((tab) => tab.value === value)
              ] = tabControl;
            }}
            onKeyDown={handleKeydown}
            onClick={handleTabChange}
            {...attributes}
            className={clsx(
              "tabs__item",
              "openapi-tabs__schema-item",
              attributes?.className as string,
              {
                active: selectedValue === value,
              }
            )}
          >
            <span className="openapi-tabs__schema-label">{label ?? value}</span>
          </li>
        ))}
      </ul>
      {showTabArrows && (
        <button
          type="button"
          className="openapi-tabs__arrow right"
          aria-label="Scroll schema tabs right"
          onClick={(event) => {
            event.preventDefault();
            tabItemListContainerRef.current!.scrollLeft += 90;
          }}
        />
      )}
    </div>
  );
}

function TabContent({
  lazy,
  children,
  selectedValue,
}: SchemaTabsProps & ReturnType<typeof useTabs>) {
  const childTabs = (Array.isArray(children) ? children : [children]).filter(
    Boolean
  ) as ReactElement<TabItemProps>[];
  const flattenedChildTabs = flatten(childTabs);

  if (lazy) {
    const selectedTabItem = flattenedChildTabs.find(
      (tabItem) => tabItem.props.value === selectedValue
    );

    if (!selectedTabItem) {
      return null;
    }

    return cloneElement(selectedTabItem, { className: "margin-top--md" });
  }

  return (
    <div className="margin-top--md">
      {childTabs.map((tabItem, index) =>
        cloneElement(tabItem, {
          key: index,
          hidden: tabItem.props.value !== selectedValue,
        })
      )}
    </div>
  );
}

function TabsComponent(props: SchemaTabsProps): React.JSX.Element {
  const tabs = useTabs();

  return (
    <div className="openapi-tabs__schema-container">
      <TabList {...props} {...tabs} />
      <TabContent {...props} {...tabs} />
    </div>
  );
}

export default function SchemaTabs(
  props: SchemaTabsProps
): React.JSX.Element {
  const isBrowser = useIsBrowser();
  const value = useTabsContextValue(props);

  return (
    <TabsProvider value={value} key={String(isBrowser)}>
      <TabsComponent {...props}>
        {sanitizeTabsChildren(props.children)}
      </TabsComponent>
    </TabsProvider>
  );
}
