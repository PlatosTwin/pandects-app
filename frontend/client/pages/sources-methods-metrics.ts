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
      id: "crf-tune-val",
      title: "Validation Metrics",
      badge: "Tuned CRF",
      layout: "accordion",
      summary: {
        accuracy: 0.9767773936569969,
        precision: 0.9663274654406944,
        recall: 0.9684444702237258,
        f1: 0.9666670514543612,
      },
      confusionMatrix: [
        [152, 1, 0, 0, 0],
        [3, 531, 2, 0, 0],
        [0, 3, 9790, 3, 244],
        [0, 0, 6, 213, 19],
        [0, 0, 23, 5, 2311],
      ],
      perClass: [
        {
          label: "front_matter",
          acc: 0.9934640522875817,
          p: 0.9806451612903225,
          r: 0.9934640522875817,
          f1: 0.987012987012987,
        },
        {
          label: "toc",
          acc: 0.9906716417910447,
          p: 0.9925233644859813,
          r: 0.9906716417910447,
          f1: 0.9915966386554622,
        },
        {
          label: "body",
          acc: 0.9750996015936255,
          p: 0.9968434986253946,
          r: 0.9750996015936255,
          f1: 0.9858516691002467,
        },
        {
          label: "sig",
          acc: 0.8949579831932774,
          p: 0.9638009049773756,
          r: 0.8949579831932774,
          f1: 0.9281045751633987,
        },
        {
          label: "back_matter",
          acc: 0.9880290722530997,
          p: 0.8978243978243978,
          r: 0.9880290722530997,
          f1: 0.9407693873397109,
        },
      ],
      matrixCaption: "Tuned CRF validation confusion matrix",
      perClassCaption: "Tuned CRF validation per-class metrics",
    },
    {
      id: "crf-final-test",
      title: "Final Test Metrics",
      badge: "Final CRF",
      layout: "card",
      summary: {
        accuracy: 0.9599875930521092,
        precision: 0.9394398872595804,
        recall: 0.9631460075097202,
        f1: 0.9507914467270675,
      },
      confusionMatrix: [
        [158, 3, 2, 0, 0],
        [4, 472, 0, 0, 0],
        [1, 5, 9005, 12, 301],
        [0, 0, 1, 167, 7],
        [0, 0, 167, 13, 2578],
      ],
      perClass: [
        {
          label: "front_matter",
          acc: 0.9693251533742331,
          p: 0.9693251533742331,
          r: 0.9693251533742331,
          f1: 0.9693251533742331,
        },
        {
          label: "toc",
          acc: 0.9915966386554622,
          p: 0.9833333333333333,
          r: 0.9915966386554622,
          f1: 0.9874476987447699,
        },
        {
          label: "body",
          acc: 0.9657872157872158,
          p: 0.9814713896457765,
          r: 0.9657872157872158,
          f1: 0.9735661387102006,
        },
        {
          label: "sig",
          acc: 0.9542857142857143,
          p: 0.8697916666666666,
          r: 0.9542857142857143,
          f1: 0.9100817438692098,
        },
        {
          label: "back_matter",
          acc: 0.9347353154459753,
          p: 0.8932778932778933,
          r: 0.9347353154459753,
          f1: 0.9135364989369241,
        },
      ],
      matrixCaption: "Final CRF test confusion matrix",
      perClassCaption: "Final CRF test per-class metrics",
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
