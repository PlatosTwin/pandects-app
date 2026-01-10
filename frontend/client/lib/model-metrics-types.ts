export type NerEvalData = {
  summary: {
    strict: { precision: number; recall: number; f1: number };
    lenient: { precision: number; recall: number; f1: number };
  };
  perEntity: {
    strict: Record<"ARTICLE" | "SECTION" | "PAGE", {
      precision: number;
      recall: number;
      f1: number;
    }>;
    lenient: Record<"ARTICLE" | "SECTION" | "PAGE", {
      precision: number;
      recall: number;
      f1: number;
    }>;
  };
  boundaries: {
    ARTICLE: { B: number; I: number; E: number };
    SECTION: { B: number; I: number; E: number };
    PAGE: { B: number; I: number; E: number; S: number };
  };
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
