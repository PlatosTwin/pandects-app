import type { ClassifierEvalData, NerEvalData } from "@/lib/model-metrics-types";

export const classifierEvalData: ClassifierEvalData = {
  labels: ["front_matter", "toc", "body", "sig", "back_matter"],
  abbreviations: ["FM", "TOC", "BDY", "SIG", "BM"],
  models: [
    {
      id: "xgb-baseline",
      title: "Model Metrics",
      badge: "XGB Baseline",
      layout: "accordion",
      summary: {
        accuracy: 0.9521843825249398,
        precision: 0.9295911238074221,
        recall: 0.9307327897623399,
        f1: 0.9270384433262334,
      },
      confusionMatrix: [
        [40, 1, 1, 1, 0],
        [0, 129, 2, 0, 0],
        [0, 0, 2171, 0, 33],
        [0, 0, 1, 49, 1],
        [0, 0, 86, 13, 379],
      ],
      perClass: [
        {
          label: "front_matter",
          acc: 0.9302325581395349,
          p: 1.0,
          r: 0.9302325581395349,
          f1: 0.963855421686747,
        },
        {
          label: "toc",
          acc: 0.9847328244274809,
          p: 0.9923076923076923,
          r: 0.9847328244274809,
          f1: 0.9885057471264368,
        },
        {
          label: "body",
          acc: 0.98502722323049,
          p: 0.9601946041574525,
          r: 0.98502722323049,
          f1: 0.9724524076147817,
        },
        {
          label: "sig",
          acc: 0.9607843137254902,
          p: 0.7777777777777778,
          r: 0.9607843137254902,
          f1: 0.8596491228070176,
        },
        {
          label: "back_matter",
          acc: 0.7928870292887029,
          p: 0.9176755447941889,
          r: 0.7928870292887029,
          f1: 0.8507295173961841,
        },
      ],
      matrixCaption: "XGB baseline confusion matrix",
      perClassCaption: "XGB baseline per-class metrics",
    },
    {
      id: "crf-final",
      title: "BiLSTM + CRF Metrics",
      badge: "BiLSTM + CRF",
      layout: "accordion",
      summary: {
        accuracy: 0.9680082559339526,
        precision: 0.9686209795317661,
        recall: 0.9295120810293076,
        f1: 0.9479381326817894,
      },
      confusionMatrix: [
        [41, 2, 0, 0, 0],
        [0, 131, 0, 0, 0],
        [0, 0, 2189, 0, 15],
        [0, 0, 3, 43, 5],
        [0, 0, 65, 3, 410],
      ],
      perClass: [
        {
          label: "front_matter",
          acc: 0.9534883720930233,
          p: 1.0,
          r: 0.9534883720930233,
          f1: 0.9761904761904762,
        },
        {
          label: "toc",
          acc: 1.0,
          p: 0.9849624060150376,
          r: 1.0,
          f1: 0.9924242424242424,
        },
        {
          label: "body",
          acc: 0.9931941923774955,
          p: 0.9698715108551174,
          r: 0.9931941923774955,
          f1: 0.9813943062093701,
        },
        {
          label: "sig",
          acc: 0.8431372549019608,
          p: 0.9347826086956522,
          r: 0.8431372549019608,
          f1: 0.8865979381443299,
        },
        {
          label: "back_matter",
          acc: 0.8577405857740585,
          p: 0.9534883720930233,
          r: 0.8577405857740585,
          f1: 0.9030837004405287,
        },
      ],
      matrixCaption: "Final classifier confusion matrix",
      perClassCaption: "Final classifier per-class metrics",
    },
    {
      id: "post-processing",
      title: "Post-processing Metrics",
      badge: "Post-processing",
      layout: "card",
      summary: {
        accuracy: 0.9728242174062608,
        precision: 0.9593666339717979,
        recall: 0.9528856145431316,
        f1: 0.9553224497648898,
      },
      confusionMatrix: [
        [41, 2, 0, 0, 0],
        [0, 131, 0, 0, 0],
        [0, 0, 2189, 0, 15],
        [0, 0, 2, 48, 1],
        [0, 0, 52, 7, 419],
      ],
      perClass: [
        {
          label: "front_matter",
          acc: 0.9534883720930233,
          p: 1.0,
          r: 0.9534883720930233,
          f1: 0.9761904761904762,
        },
        {
          label: "toc",
          acc: 1.0,
          p: 0.9849624060150376,
          r: 1.0,
          f1: 0.9924242424242424,
        },
        {
          label: "body",
          acc: 0.9931941923774955,
          p: 0.975925100312082,
          r: 0.9931941923774955,
          f1: 0.9844839217449967,
        },
        {
          label: "sig",
          acc: 0.9411764705882353,
          p: 0.8727272727272727,
          r: 0.9411764705882353,
          f1: 0.9056603773584906,
        },
        {
          label: "back_matter",
          acc: 0.8765690376569037,
          p: 0.9632183908045977,
          r: 0.8765690376569037,
          f1: 0.9178532311062432,
        },
      ],
      matrixCaption: "Post-processing confusion matrix",
      perClassCaption: "Post-processing per-class metrics",
    },
  ],
};

export const nerEvalData: NerEvalData = {
  meta: {
    splitVersion: "v1.2",
    labelScheme: "BIOE",
    finalTrainDocs: 7000,
    finalArticleWeight: 3,
    finalGatingMode: "regex+snap",
  },
  finalTest: {
    primaryMode: "regex+snap",
    summaryByMode: {
      raw: {
        entityStrict: { precision: 0.91, recall: 0.88, f1: 0.895 },
        entityLenient: { f1: 0.94 },
        articleStrict: { precision: 0.86, recall: 0.82, f1: 0.84 },
        accuracy: 0.93,
      },
      regex: {
        entityStrict: { precision: 0.93, recall: 0.9, f1: 0.915 },
        entityLenient: { f1: 0.95 },
        articleStrict: { precision: 0.88, recall: 0.84, f1: 0.86 },
        accuracy: 0.94,
      },
      "regex+snap": {
        entityStrict: { precision: 0.95, recall: 0.92, f1: 0.935 },
        entityLenient: { f1: 0.965 },
        articleStrict: { precision: 0.9, recall: 0.87, f1: 0.885 },
        accuracy: 0.956,
      },
    },
    perTypeStrict: [
      { type: "PAGE", precision: 0.96, recall: 0.93, f1: 0.945, support: 3200 },
      {
        type: "SECTION",
        precision: 0.93,
        recall: 0.9,
        f1: 0.915,
        support: 1800,
      },
      { type: "ARTICLE", precision: 0.9, recall: 0.86, f1: 0.88, support: 740 },
    ],
  },
  baselineVal: {
    byMode: {
      raw: {
        entityStrict: { precision: 0.88, recall: 0.84, f1: 0.86 },
        entityLenient: { f1: 0.91 },
        articleStrict: { precision: 0.8, recall: 0.75, f1: 0.775 },
      },
      regex: {
        entityStrict: { precision: 0.9, recall: 0.86, f1: 0.88 },
        entityLenient: { f1: 0.925 },
        articleStrict: { precision: 0.82, recall: 0.78, f1: 0.8 },
      },
      "regex+snap": {
        entityStrict: { precision: 0.91, recall: 0.88, f1: 0.895 },
        entityLenient: { f1: 0.935 },
        articleStrict: { precision: 0.84, recall: 0.8, f1: 0.82 },
      },
    },
  },
  learningCurveVal: [
    {
      trainDocs: 1500,
      byMode: {
        raw: { entityStrictF1: 0.78, articleStrictF1: 0.62 },
        regex: { entityStrictF1: 0.81, articleStrictF1: 0.66 },
        "regex+snap": { entityStrictF1: 0.83, articleStrictF1: 0.68 },
      },
    },
    {
      trainDocs: 3500,
      byMode: {
        raw: { entityStrictF1: 0.82, articleStrictF1: 0.68 },
        regex: { entityStrictF1: 0.85, articleStrictF1: 0.71 },
        "regex+snap": { entityStrictF1: 0.88, articleStrictF1: 0.74 },
      },
    },
    {
      trainDocs: 7000,
      byMode: {
        raw: { entityStrictF1: 0.86, articleStrictF1: 0.73 },
        regex: { entityStrictF1: 0.89, articleStrictF1: 0.76 },
        "regex+snap": { entityStrictF1: 0.915, articleStrictF1: 0.79 },
      },
    },
  ],
  weightSweepVal: [
    {
      articleWeight: 1,
      byMode: {
        raw: { entityStrictF1: 0.86, articleStrictF1: 0.74 },
        regex: { entityStrictF1: 0.88, articleStrictF1: 0.76 },
        "regex+snap": { entityStrictF1: 0.9, articleStrictF1: 0.78 },
      },
    },
    {
      articleWeight: 2,
      byMode: {
        raw: { entityStrictF1: 0.87, articleStrictF1: 0.76 },
        regex: { entityStrictF1: 0.89, articleStrictF1: 0.79 },
        "regex+snap": { entityStrictF1: 0.915, articleStrictF1: 0.82 },
      },
    },
    {
      articleWeight: 3,
      byMode: {
        raw: { entityStrictF1: 0.88, articleStrictF1: 0.78 },
        regex: { entityStrictF1: 0.91, articleStrictF1: 0.83 },
        "regex+snap": { entityStrictF1: 0.935, articleStrictF1: 0.86 },
      },
    },
  ],
};
