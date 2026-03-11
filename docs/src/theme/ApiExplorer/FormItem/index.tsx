import React from "react";

import { translate } from "@docusaurus/Translate";
import { OPENAPI_SCHEMA_ITEM } from "@theme/translationIds";
import clsx from "clsx";

export interface Props {
  label?: string;
  type?: string;
  required?: boolean | undefined;
  pending?: boolean | undefined;
  children?: React.ReactNode;
  className?: string;
}

function FormItem({
  label,
  type,
  required,
  pending,
  children,
  className,
}: Props) {
  const meta = (label || type || required || pending) && (
    <div className="openapi-schema__container openapi-explorer__form-item-meta">
      {label && (
        <span className="openapi-explorer__form-item-label openapi-schema__property">
          {label}
        </span>
      )}
      {type && <span className="openapi-schema__type"> {"\u2014"} {type}</span>}
      {(required || pending) && (
        <>
          <span className="openapi-schema__divider"></span>
          {required && (
            <span className="openapi-schema__required">
              {translate({
                id: OPENAPI_SCHEMA_ITEM.REQUIRED,
                message: "required",
              })}
            </span>
          )}
          {pending && <span className="openapi-schema__pending">Pending</span>}
        </>
      )}
    </div>
  );

  return (
    <div className={clsx("openapi-explorer__form-item", className)}>
      {label ? (
        <label>
          {meta}
          <div>{children}</div>
        </label>
      ) : (
        <>
          {meta}
          <div>{children}</div>
        </>
      )}
    </div>
  );
}

export default FormItem;
