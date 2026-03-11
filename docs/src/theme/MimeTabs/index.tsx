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
  useScrollPositionBlocker,
  useTabs,
} from "@docusaurus/theme-common/internal";
import { TabItemProps } from "@docusaurus/theme-common/lib/utils/tabsUtils";
import useIsBrowser from "@docusaurus/useIsBrowser";
import { setAccept } from "@theme/ApiExplorer/Accept/slice";
import { setContentType } from "@theme/ApiExplorer/ContentType/slice";
import { useTypedDispatch, useTypedSelector } from "@theme/ApiItem/hooks";
import { RootState } from "@theme/ApiItem/store";
import clsx from "clsx";

export interface Props {
  schemaType: string;
}

function TabList({
  className,
  block,
  selectedValue: selectedValueProp,
  selectValue,
  tabValues,
  schemaType,
}: Props & TabProps & ReturnType<typeof useTabs>): React.JSX.Element {
  const tabRefs = useRef<(HTMLLIElement | null)[]>([]);
  const tabItemListContainerRef = useRef<HTMLUListElement>(null);
  const [showTabArrows, setShowTabArrows] = useState(false);
  const [selectedValue, setSelectedValue] = useState(selectedValueProp);
  const { blockElementScrollPositionUntilNextRender } =
    useScrollPositionBlocker();

  const dispatch = useTypedDispatch();
  const isRequestSchema = schemaType?.toLowerCase() === "request";
  const contentTypeValue = useTypedSelector(
    (state: RootState) => state.contentType.value
  );
  const acceptTypeValue = useTypedSelector(
    (state: RootState) => state.accept.value
  );

  useEffect(() => {
    if (tabRefs.current.length <= 1) {
      return;
    }

    const nextSelectedValue = isRequestSchema
      ? contentTypeValue
      : acceptTypeValue;

    if (!nextSelectedValue) {
      return;
    }

    setSelectedValue(nextSelectedValue);
  }, [acceptTypeValue, contentTypeValue, isRequestSchema]);

  const handleTabChange = (
    event:
      | React.FocusEvent<HTMLLIElement>
      | React.MouseEvent<HTMLLIElement>
      | React.KeyboardEvent<HTMLLIElement>
  ) => {
    event.preventDefault();

    const newTab = event.currentTarget;
    const newTabIndex = tabRefs.current.indexOf(newTab);
    const newTabValue = tabValues[newTabIndex]!.value;

    if (newTabValue !== selectedValue) {
      if (isRequestSchema) {
        dispatch(setContentType(newTabValue));
      } else {
        dispatch(setAccept(newTabValue));
      }

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

    resizeObserver.observe(tabItemListContainerRef.current!);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <div className="tabs__container">
      <div className="openapi-tabs__mime-container">
        {showTabArrows && (
          <button
            type="button"
            className={clsx("openapi-tabs__arrow", "left")}
            aria-label="Scroll content type tabs left"
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
            "openapi-tabs__mime-list-container",
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
                tabRefs.current.push(tabControl);
              }}
              onKeyDown={handleKeydown}
              onFocus={handleTabChange}
              onClick={handleTabChange}
              {...attributes}
              className={clsx(
                "tabs__item",
                "openapi-tabs__mime-item",
                attributes?.className as string,
                {
                  active: selectedValue === value,
                }
              )}
            >
              {label ?? value}
            </li>
          ))}
        </ul>
        {showTabArrows && (
          <button
            type="button"
            className={clsx("openapi-tabs__arrow", "right")}
            aria-label="Scroll content type tabs right"
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
}: Props & TabProps & ReturnType<typeof useTabs>) {
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

function TabsComponent(props: Props & TabProps): React.JSX.Element {
  const tabs = useTabs(props);

  return (
    <div className="tabs-container">
      <TabList {...props} {...tabs} />
      <TabContent {...props} {...tabs} />
    </div>
  );
}

export default function MimeTabs(
  props: Props & TabProps
): React.JSX.Element {
  const isBrowser = useIsBrowser();

  return (
    <TabsComponent key={String(isBrowser)} {...props}>
      {sanitizeTabsChildren(props.children)}
    </TabsComponent>
  );
}
