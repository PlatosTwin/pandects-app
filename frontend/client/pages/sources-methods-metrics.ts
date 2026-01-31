import type {
  ClassifierEvalData,
  NerEvalData,
  ExhibitEvalData,
} from "@/lib/model-metrics-types";

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
        accuracy: 0.9596214511041009,
        precision: 0.9372932865631419,
        recall: 0.9520630015086425,
        f1: 0.9433498592558431,
      },
      confusionMatrix: [
        [44, 1, 1, 1, 0],
        [0, 156, 0, 0, 0],
        [0, 0, 2364, 0, 59],
        [0, 0, 0, 70, 1],
        [0, 0, 53, 12, 408],
      ],
      perClass: [
        {
          label: "front_matter",
          acc: 0.9361702127659575,
          p: 1.0,
          r: 0.9361702127659575,
          f1: 0.967032967032967,
        },
        {
          label: "toc",
          acc: 1.0,
          p: 0.9936305732484076,
          r: 1.0,
          f1: 0.9968051118210862,
        },
        {
          label: "body",
          acc: 0.9756500206355757,
          p: 0.9776674937965261,
          r: 0.9756500206355757,
          f1: 0.9766577153480686,
        },
        {
          label: "sig",
          acc: 0.9859154929577465,
          p: 0.8433734939759037,
          r: 0.9859154929577465,
          f1: 0.9090909090909091,
        },
        {
          label: "back_matter",
          acc: 0.8625792811839323,
          p: 0.8717948717948718,
          r: 0.8625792811839323,
          f1: 0.8671625929861849,
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
        accuracy: 0.9842271293375394,
        precision: 0.9809909950836386,
        recall: 0.9652950810168649,
        f1: 0.9727222114209235,
      },
      confusionMatrix: [
        [44, 3, 0, 0, 0],
        [0, 156, 0, 0, 0],
        [0, 0, 2415, 0, 8],
        [0, 0, 1, 69, 1],
        [0, 0, 34, 3, 436],
      ],
      perClass: [
        {
          label: "front_matter",
          acc: 0.9361702127659575,
          p: 1.0,
          r: 0.9361702127659575,
          f1: 0.967032967032967,
        },
        {
          label: "toc",
          acc: 1.0,
          p: 0.9811320754716981,
          r: 1.0,
          f1: 0.9904761904761905,
        },
        {
          label: "body",
          acc: 0.9966983078827899,
          p: 0.9857142857142858,
          r: 0.9966983078827899,
          f1: 0.9911758670223682,
        },
        {
          label: "sig",
          acc: 0.971830985915493,
          p: 0.9583333333333334,
          r: 0.971830985915493,
          f1: 0.965034965034965,
        },
        {
          label: "back_matter",
          acc: 0.9217758985200846,
          p: 0.9797752808988764,
          r: 0.9217758985200846,
          f1: 0.9498910675381264,
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
        accuracy: 0.9842271293375394,
        precision: 0.9809909950836386,
        recall: 0.9652950810168649,
        f1: 0.9727222114209235,
      },
      confusionMatrix: [
        [44, 3, 0, 0, 0],
        [0, 156, 0, 0, 0],
        [0, 0, 2415, 0, 8],
        [0, 0, 1, 69, 1],
        [0, 0, 34, 3, 436],
      ],
      perClass: [
        {
          label: "front_matter",
          acc: 0.9361702127659575,
          p: 1.0,
          r: 0.9361702127659575,
          f1: 0.967032967032967,
        },
        {
          label: "toc",
          acc: 1.0,
          p: 0.9811320754716981,
          r: 1.0,
          f1: 0.9904761904761905,
        },
        {
          label: "body",
          acc: 0.9966983078827899,
          p: 0.9857142857142858,
          r: 0.9966983078827899,
          f1: 0.9911758670223682,
        },
        {
          label: "sig",
          acc: 0.971830985915493,
          p: 0.9583333333333334,
          r: 0.971830985915493,
          f1: 0.965034965034965,
        },
        {
          label: "back_matter",
          acc: 0.9217758985200846,
          p: 0.9797752808988764,
          r: 0.9217758985200846,
          f1: 0.9498910675381264,
        },
      ],
      matrixCaption: "Post-processing confusion matrix",
      perClassCaption: "Post-processing per-class metrics",
    },
  ],
};

export const nerEvalData: NerEvalData = {
  summary: {
    strict: {
      precision: 0.9802739726027397,
      recall: 0.9829670329670329,
      f1: 0.9816186556927298,
    },
    lenient: {
      precision: 0.993972602739726,
      recall: 0.9967032967032967,
      f1: 0.9953360768175583,
    },
  },
  perEntity: {
    strict: {
      ARTICLE: {
        precision: 0.8645833333333334,
        recall: 0.8736842105263158,
        f1: 0.8691099476439791,
      },
      SECTION: {
        precision: 0.9763617677286742,
        recall: 0.981404958677686,
        f1: 0.9788768675940237,
      },
      PAGE: {
        precision: 1.0,
        recall: 0.9986789960369881,
        f1: 0.9993390614672836,
      },
    },
    lenient: {
      ARTICLE: {
        precision: 0.9895833333333334,
        recall: 1.0,
        f1: 0.9947643979057591,
      },
      SECTION: {
        precision: 0.9897225077081192,
        recall: 0.9948347107438017,
        f1: 0.9922720247295209,
      },
      PAGE: {
        precision: 1.0,
        recall: 0.9986789960369881,
        f1: 0.9993390614672836,
      },
    },
  },
  boundaries: {
    ARTICLE: {
      B: 0.9947643979057591,
      I: 0.9848484848484849,
      E: 0.8691099476439791,
    },
    SECTION: {
      B: 0.9922720247295209,
      I: 0.9927652733118971,
      E: 0.9788768675940237,
    },
    PAGE: {
      B: 0.996078431372549,
      I: 1.0,
      E: 0.996078431372549,
      S: 1.0,
    },
  },
};

export const exhibitEvalData: ExhibitEvalData = {
  summary: {
    accuracy: 0.9647058823529412,
    precision: 0.9962962962962963,
    recall: 0.9607142857142857,
    f1: 0.9781818181818182,
    roc_auc: 0.9870238095238095,
  },
  confusionMatrix: [
    [59, 1],
    [11, 269],
  ],
  perClass: [
    {
      label: "class_0",
      accuracy: 0.9833333333333333,
      precision: 0.8428571428571429,
      recall: 0.9833333333333333,
      f1: 0.9076923076923077,
    },
    {
      label: "class_1",
      accuracy: 0.9607142857142857,
      precision: 0.9962962962962963,
      recall: 0.9607142857142857,
      f1: 0.9781818181818182,
    },
  ],
};
