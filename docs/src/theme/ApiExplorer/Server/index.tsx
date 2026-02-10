import React, { useEffect, useMemo, useRef, useState } from "react";

import { translate } from "@docusaurus/Translate";
import FloatingButton from "@theme/ApiExplorer/FloatingButton";
import FormItem from "@theme/ApiExplorer/FormItem";
import FormSelect from "@theme/ApiExplorer/FormSelect";
import FormTextInput from "@theme/ApiExplorer/FormTextInput";
import { setServer, setServerVariable } from "@theme/ApiExplorer/Server/slice";
import { useTypedDispatch, useTypedSelector } from "@theme/ApiItem/hooks";
import { OPENAPI_SERVER } from "@theme/translationIds";

const LOCAL_API_URL = "http://localhost:5113";
const PROD_API_URL = "https://api.pandects.org";

function isLocalHostname(hostname: string): boolean {
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname.endsWith(".local")
  );
}

function normalizeServerOptions(options: any[]): any[] {
  const byUrl = new Map<string, any>();

  options.forEach((option) => {
    if (!option?.url) {
      return;
    }
    byUrl.set(option.url, option);
  });

  if (!byUrl.has(LOCAL_API_URL)) {
    byUrl.set(LOCAL_API_URL, {
      url: LOCAL_API_URL,
      description: "Local development API",
    });
  }

  if (!byUrl.has(PROD_API_URL)) {
    byUrl.set(PROD_API_URL, {
      url: PROD_API_URL,
      description: "Production API",
    });
  }

  return Array.from(byUrl.values());
}

function renderResolvedUrl(server: any): string {
  if (!server?.url) {
    return "";
  }

  let url = server.url.replace(/\/$/, "");
  if (server.variables) {
    Object.keys(server.variables).forEach((variable) => {
      url = url.replace(
        `{${variable}}`,
        server.variables?.[variable].default ?? ""
      );
    });
  }

  return url;
}

function Server() {
  const [isEditing, setIsEditing] = useState(false);
  const value = useTypedSelector((state: any) => state.server.value);
  const options = useTypedSelector((state: any) => state.server.options);
  const dispatch = useTypedDispatch();
  const hasInitializedRef = useRef(false);

  const normalizedOptions = useMemo(() => {
    return normalizeServerOptions(options ?? []);
  }, [options]);

  const activeServer = useMemo(() => {
    if (!value?.url) {
      return undefined;
    }

    return normalizedOptions.find((option) => option.url === value.url) ?? value;
  }, [normalizedOptions, value]);

  useEffect(() => {
    if (hasInitializedRef.current || normalizedOptions.length === 0) {
      return;
    }

    const defaultUrl = isLocalHostname(window.location.hostname)
      ? LOCAL_API_URL
      : PROD_API_URL;

    const defaultServer =
      normalizedOptions.find((option) => option.url === defaultUrl) ??
      normalizedOptions[0];

    dispatch(setServer(JSON.stringify(defaultServer)));
    hasInitializedRef.current = true;
  }, [dispatch, normalizedOptions]);

  if (normalizedOptions.length <= 0) {
    return null;
  }

  if (!isEditing) {
    return (
      <FloatingButton
        onClick={() => setIsEditing(true)}
        label={translate({ id: OPENAPI_SERVER.EDIT_BUTTON, message: "Edit" })}
      >
        <FormItem>
          <span
            className="openapi-explorer__server-url"
            title={renderResolvedUrl(activeServer)}
          >
            {renderResolvedUrl(activeServer)}
          </span>
        </FormItem>
      </FloatingButton>
    );
  }

  return (
    <div className="openapi-explorer__server-container">
      <FloatingButton
        onClick={() => setIsEditing(false)}
        label={translate({ id: OPENAPI_SERVER.HIDE_BUTTON, message: "Hide" })}
      >
        <FormItem>
          <FormSelect
            options={normalizedOptions.map((option) => option.url)}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
              const selected = normalizedOptions.find(
                (option) => option.url === e.target.value
              );
              if (!selected) {
                return;
              }
              dispatch(setServer(JSON.stringify(selected)));
            }}
            value={activeServer?.url}
          />
          <small className="openapi-explorer__server-description">
            {activeServer?.description}
          </small>
        </FormItem>
        {activeServer?.variables &&
          Object.keys(activeServer.variables).map((key) => {
            const variable = activeServer.variables?.[key];
            if (variable?.enum !== undefined) {
              return (
                <FormItem key={key} label={key}>
                  <FormSelect
                    options={variable.enum}
                    onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
                      dispatch(
                        setServerVariable(
                          JSON.stringify({ key, value: e.target.value })
                        )
                      );
                    }}
                    value={variable.default}
                  />
                </FormItem>
              );
            }

            return (
              <FormItem key={key} label={key}>
                <FormTextInput
                  placeholder={variable?.default}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                    dispatch(
                      setServerVariable(
                        JSON.stringify({ key, value: e.target.value })
                      )
                    );
                  }}
                  value={variable?.default}
                />
              </FormItem>
            );
          })}
      </FloatingButton>
    </div>
  );
}

export default Server;
