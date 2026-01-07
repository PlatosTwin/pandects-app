export type NerEvalData = {
  meta: {
    splitVersion: string;
    labelScheme: string;
    finalTrainDocs: number;
    finalArticleWeight: number;
    finalGatingMode: "raw" | "regex" | "regex+snap";
  };
  finalTest: {
    primaryMode: "raw" | "regex" | "regex+snap";
    summaryByMode: Record<
      "raw" | "regex" | "regex+snap",
      {
        entityStrict: { precision: number; recall: number; f1: number };
        entityLenient: { f1: number };
        articleStrict?: { precision: number; recall: number; f1: number };
        accuracy?: number;
      }
    >;
    perTypeStrict: Array<{
      type: "PAGE" | "SECTION" | "ARTICLE";
      precision: number;
      recall: number;
      f1: number;
      support: number;
    }>;
  };
  baselineVal: {
    byMode: Record<
      "raw" | "regex" | "regex+snap",
      {
        entityStrict: { precision: number; recall: number; f1: number };
        entityLenient: { f1: number };
        articleStrict?: { precision: number; recall: number; f1: number };
      }
    >;
  };
  learningCurveVal: Array<{
    trainDocs: number;
    byMode: Record<
      "raw" | "regex" | "regex+snap",
      {
        entityStrictF1: number;
        articleStrictF1?: number;
      }
    >;
  }>;
  weightSweepVal: Array<{
    articleWeight: number;
    byMode: Record<
      "raw" | "regex" | "regex+snap",
      {
        entityStrictF1: number;
        articleStrictF1: number;
      }
    >;
  }>;
};

export type ClassifierEvalData = {
  labels: string[];
  abbreviations: string[];
  models: Array<{
    id: string;
    title: string;
    badge: string;
    layout: "accordion" | "card";
    summary: {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
    };
    confusionMatrix: number[][];
    perClass: Array<{
      label: string;
      acc: number;
      p: number;
      r: number;
      f1: number;
    }>;
    matrixCaption: string;
    perClassCaption: string;
  }>;
};

export type ExhibitEvalData = {
  summary: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
  };
  confusionMatrix: number[][];
  perClass: Array<{
    label: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  }>;
};
