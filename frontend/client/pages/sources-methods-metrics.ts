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
  summary: {
    strict: {
      precision: 0.9813504823151126,
      recall: 0.9851517107811492,
      f1: 0.9832474226804123,
    },
    lenient: {
      precision: 0.992282958199357,
      recall: 0.9961265332472563,
      f1: 0.9942010309278351,
    },
  },
  perEntity: {
    strict: {
      ARTICLE: {
        precision: 0.825,
        recall: 0.8148148148148148,
        f1: 0.8198757763975155,
      },
      SECTION: {
        precision: 0.9845605700712589,
        recall: 0.9928143712574851,
        f1: 0.9886702444841979,
      },
      PAGE: {
        precision: 0.9968404423380727,
        recall: 0.9968404423380727,
        f1: 0.9968404423380727,
      },
    },
    lenient: {
      ARTICLE: {
        precision: 1.0,
        recall: 0.9876543209876543,
        f1: 0.9937888198757764,
      },
      SECTION: {
        precision: 0.9881235154394299,
        recall: 0.9964071856287425,
        f1: 0.992248062015504,
      },
      PAGE: {
        precision: 0.9968404423380727,
        recall: 0.9968404423380727,
        f1: 0.9968404423380727,
      },
    },
  },
  boundaries: {
    ARTICLE: {
      B: 0.9937888198757764,
      I: 0.9796355841371919,
      E: 0.8198757763975155,
    },
    SECTION: {
      B: 0.992248062015504,
      I: 0.9956439393939395,
      E: 0.9886702444841979,
    },
    PAGE: {
      B: 0.9965635738831615,
      I: 0.996415770609319,
      E: 0.9965635738831615,
      S: 0.9969230769230769,
    },
  },
};

export const exhibitEvalData: ExhibitEvalData = {
  summary: {
    accuracy: 0.9756838905775076,
    precision: 0.9927272727272727,
    recall: 0.978494623655914,
    f1: 0.9855595667870036,
    roc_auc: 0.9864516129032258,
  },
  confusionMatrix: [
    [48, 2],
    [6, 273],
  ],
  perClass: [
    {
      label: "class_0",
      accuracy: 0.96,
      precision: 0.8888888888888888,
      recall: 0.96,
      f1: 0.9230769230769231,
    },
    {
      label: "class_1",
      accuracy: 0.978494623655914,
      precision: 0.9927272727272727,
      recall: 0.978494623655914,
      f1: 0.9855595667870036,
    },
  ],
};
