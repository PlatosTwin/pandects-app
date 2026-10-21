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
      id: "crf-typesafe-val",
      title: "Validation Metrics",
      badge: "CRF + TypeSafe",
      layout: "accordion",
      summary: {
        accuracy: 0.9815817109621132,
        precision: 0.9666234447247735,
        recall: 0.977302155826002,
        f1: 0.9717501480803046,
      },
      confusionMatrix: [
        [139, 1, 0, 0, 0],
        [5, 480, 8, 0, 0],
        [0, 0, 9016, 8, 156],
        [0, 0, 0, 192, 9],
        [0, 0, 37, 4, 2324],
      ],
      perClass: [
        {
          label: "front_matter",
          acc: 0.9928571428571429,
          p: 0.9652777777777778,
          r: 0.9928571428571429,
          f1: 0.9788732394366197,
        },
        {
          label: "toc",
          acc: 0.973630831643002,
          p: 0.997920997920998,
          r: 0.973630831643002,
          f1: 0.9856262833675564,
        },
        {
          label: "body",
          acc: 0.9821350762527233,
          p: 0.9950336607438472,
          r: 0.9821350762527233,
          f1: 0.9885422948303273,
        },
        {
          label: "sig",
          acc: 0.9552238805970149,
          p: 0.9411764705882353,
          r: 0.9552238805970149,
          f1: 0.9481481481481482,
        },
        {
          label: "back_matter",
          acc: 0.9826638477801268,
          p: 0.9337083165930092,
          r: 0.9826638477801268,
          f1: 0.957560774618871,
        },
      ],
      matrixCaption: "CRF + TypeSafe validation confusion matrix",
      perClassCaption: "CRF + TypeSafe validation per-class metrics",
    },
    {
      id: "crf-typesafe-final-test",
      title: "Final Test Metrics",
      badge: "Final CRF + TypeSafe",
      layout: "card",
      summary: {
        accuracy: 0.9967031060211695,
        precision: 0.9884518410138365,
        recall: 0.992414325340019,
        f1: 0.9903998045841298,
      },
      confusionMatrix: [
        [138, 1, 0, 0, 0],
        [3, 408, 0, 0, 0],
        [0, 0, 8022, 1, 0],
        [0, 0, 0, 162, 2],
        [0, 0, 27, 4, 2758],
      ],
      perClass: [
        {
          label: "front_matter",
          acc: 0.9928057553956835,
          p: 0.9787234042553191,
          r: 0.9928057553956835,
          f1: 0.9857142857142858,
        },
        {
          label: "toc",
          acc: 0.9927007299270073,
          p: 0.9975550122249389,
          r: 0.9927007299270073,
          f1: 0.9951219512195122,
        },
        {
          label: "body",
          acc: 0.9998753583447588,
          p: 0.9966455460305628,
          r: 0.9998753583447588,
          f1: 0.9982578397212544,
        },
        {
          label: "sig",
          acc: 0.9878048780487805,
          p: 0.9700598802395209,
          r: 0.9878048780487805,
          f1: 0.9788519637462235,
        },
        {
          label: "back_matter",
          acc: 0.9888849049838652,
          p: 0.9992753623188406,
          r: 0.9888849049838652,
          f1: 0.9940529825193729,
        },
      ],
      matrixCaption: "Final CRF + TypeSafe test confusion matrix",
      perClassCaption: "Final CRF + TypeSafe test per-class metrics",
    },
  ],
};

export const nerEvalData: NerEvalData = {
  summary: {
    strict: {
      precision: 0.9981962481962482,
      recall: 0.9989169675090253,
      f1: 0.9985564778058463,
    },
    lenient: {
      precision: 0.9992784992784993,
      recall: 1.0,
      f1: 0.9996391194514616,
    },
  },
  perEntity: {
    strict: {
      ARTICLE: {
        precision: 1.0,
        recall: 1.0,
        f1: 1.0,
      },
      SECTION: {
        precision: 0.9982502187226596,
        recall: 0.9986870897155361,
        f1: 0.9984686064318529,
      },
      PAGE: {
        precision: 0.9978749241044323,
        recall: 0.9990881458966565,
        f1: 0.9984811664641555,
      },
    },
    lenient: {
      ARTICLE: {
        precision: 1.0,
        recall: 1.0,
        f1: 1.0,
      },
      SECTION: {
        precision: 0.9995625546806649,
        recall: 1.0,
        f1: 0.9997812294902646,
      },
      PAGE: {
        precision: 0.9987856709168185,
        recall: 1.0,
        f1: 0.9993924665856623,
      },
    },
  },
  boundaries: {
    ARTICLE: {
      B: 1.0,
      I: 1.0,
      E: 1.0,
    },
    SECTION: {
      B: 0.9997812294902646,
      I: 0.9994131455399061,
      E: 0.9986873769415884,
    },
    PAGE: {
      B: 0.9971883786316776,
      I: 0.9971883786316776,
      E: 0.9971883786316776,
      S: 0.9987311944897589,
    },
  },
};

export const exhibitEvalData: ExhibitEvalData = {
  summary: {
    accuracy: 0.9723788049605412,
    precision: 0.9974009096816114,
    recall: 0.9715189873417721,
    f1: 0.9842898364860532,
    roc_auc: 0.9960948714602635,
  },
  confusionMatrix: [
    [190, 4],
    [45, 1535],
  ],
  perClass: [
    {
      label: "class_0",
      accuracy: 0.979381443298969,
      precision: 0.8085106382978723,
      recall: 0.979381443298969,
      f1: 0.8857808857808858,
    },
    {
      label: "class_1",
      accuracy: 0.9715189873417721,
      precision: 0.9974009096816114,
      recall: 0.9715189873417721,
      f1: 0.9842898364860532,
    },
  ],
};
