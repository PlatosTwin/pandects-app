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

import Heading from "@theme/Heading";
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

function TabList({
  label,
  id,
  className,
  block,
  selectedValue,
  selectValue,
  tabValues,
}: TabProps & ReturnType<typeof useTabs>) {
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
    <div className="openapi-tabs__response-header-section">
      <Heading
        as="h2"
        id={id}
        className="openapi-tabs__heading openapi-tabs__response-header"
      >
        {label}
      </Heading>
      <div className="openapi-tabs__response-container">
        {showTabArrows && (
          <button
            type="button"
            className="openapi-tabs__arrow left"
            aria-label="Scroll response tabs left"
            onClick={() => {
              tabItemListContainerRef.current!.scrollLeft -= 90;
            }}
          />
        )}
        <ul
          ref={tabItemListContainerRef}
          role="tablist"
          aria-orientation="horizontal"
          className={clsx(
            "openapi-tabs__response-list-container",
            "tabs",
            {
              "tabs--block": block,
            },
            className
          )}
        >
          {tabValues.map(({ value, label: tabLabel, attributes }) => (
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
                "openapi-tabs__response-code-item",
                attributes?.className as string,
                Number.parseInt(value, 10) >= 400
                  ? "danger"
                  : Number.parseInt(value, 10) >= 200 &&
                      Number.parseInt(value, 10) < 300
                    ? "success"
                    : "info",
                {
                  active: selectedValue === value,
                }
              )}
            >
              {tabLabel ?? value}
            </li>
          ))}
        </ul>
        {showTabArrows && (
          <button
            type="button"
            className="openapi-tabs__arrow right"
            aria-label="Scroll response tabs right"
            onClick={() => {
              tabItemListContainerRef.current!.scrollLeft += 90;
            }}
          />
        )}
      </div>
    </div>
  );
}

function TabContent({
  lazy,
  children,
  selectedValue,
}: TabProps & ReturnType<typeof useTabs>): React.JSX.Element | null {
  const childTabs = (Array.isArray(children) ? children : [children]).filter(
    Boolean
  ) as ReactElement<TabItemProps>[];

  if (lazy) {
    const selectedTabItem = childTabs.find(
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

function TabsComponent(props: TabProps): React.JSX.Element {
  const tabs = useTabs();

  return (
    <div className="openapi-tabs__container">
      <TabList {...props} {...tabs} />
      <TabContent {...props} {...tabs} />
    </div>
  );
}

export default function ApiTabs(props: TabProps): React.JSX.Element {
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
