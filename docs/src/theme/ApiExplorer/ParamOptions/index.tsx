import React from "react";

import FormItem from "@theme/ApiExplorer/FormItem";
import ParamArrayFormItem from "@theme/ApiExplorer/ParamOptions/ParamFormItems/ParamArrayFormItem";
import ParamBooleanFormItem from "@theme/ApiExplorer/ParamOptions/ParamFormItems/ParamBooleanFormItem";
import ParamMultiSelectFormItem from "@theme/ApiExplorer/ParamOptions/ParamFormItems/ParamMultiSelectFormItem";
import ParamSelectFormItem from "@theme/ApiExplorer/ParamOptions/ParamFormItems/ParamSelectFormItem";
import ParamTextFormItem from "@theme/ApiExplorer/ParamOptions/ParamFormItems/ParamTextFormItem";
import { useTypedSelector } from "@theme/ApiItem/hooks";

import { Param } from "@theme/ApiExplorer/ParamOptions/slice";

interface ParamProps {
  param: Param;
}

function ParamOption({ param }: ParamProps) {
  if (param.schema?.type === "array" && param.schema.items?.enum) {
    return <ParamMultiSelectFormItem param={param} />;
  }

  if (param.schema?.type === "array") {
    return <ParamArrayFormItem param={param} />;
  }

  if (param.schema?.enum) {
    return <ParamSelectFormItem param={param} />;
  }

  if (param.schema?.type === "boolean") {
    return <ParamBooleanFormItem param={param} />;
  }

  return <ParamTextFormItem param={param} />;
}

function ParamOptionWrapper({ param }: ParamProps) {
  return (
    <FormItem label={param.name} type={param.in} required={param.required}>
      <ParamOption param={param} />
    </FormItem>
  );
}

export default function ParamOptions() {
  const pathParams = useTypedSelector((state: any) => state.params.path);
  const queryParams = useTypedSelector((state: any) => state.params.query);
  const cookieParams = useTypedSelector((state: any) => state.params.cookie);
  const headerParams = useTypedSelector((state: any) => state.params.header);

  const allParams = [
    ...pathParams,
    ...queryParams,
    ...cookieParams,
    ...headerParams,
  ];
  const requiredParams = allParams.filter((p) => p.required);
  const optionalParams = allParams.filter((p) => !p.required);

  return (
    <>
      {requiredParams.map((param) => (
        <ParamOptionWrapper key={`${param.in}-${param.name}`} param={param} />
      ))}

      {optionalParams.length > 0 && (
        <div className="openapi-explorer__show-options">
          {optionalParams.map((param) => (
            <ParamOptionWrapper key={`${param.in}-${param.name}`} param={param} />
          ))}
        </div>
      )}
    </>
  );
}
