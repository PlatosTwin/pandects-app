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
      precision: 0.9722013523666416,
      recall: 0.9683212771264654,
      f1: 0.9702574356410897,
    },
    lenient: {
      precision: 0.9809666917104933,
      recall: 0.9770516338238963,
      f1: 0.979005248687828,
    },
  },
  perEntity: {
    strict: {
      ARTICLE: {
        precision: 0.9871794871794872,
        recall: 0.9829787234042553,
        f1: 0.9850746268656716,
      },
      SECTION: {
        precision: 0.9967682363804248,
        recall: 0.9986123959296948,
        f1: 0.9976894639556377,
      },
      PAGE: {
        precision: 0.9365976145637163,
        recall: 0.9255583126550868,
        f1: 0.9310452418096723,
      },
    },
    lenient: {
      ARTICLE: {
        precision: 0.9914529914529915,
        recall: 0.9872340425531915,
        f1: 0.9893390191897654,
      },
      SECTION: {
        precision: 0.9976915974145891,
        recall: 0.9995374653098983,
        f1: 0.9986136783733828,
      },
      PAGE: {
        precision: 0.9566854990583804,
        recall: 0.9454094292803971,
        f1: 0.9510140405616225,
      },
    },
  },
  boundaries: {
    ARTICLE: {
      B: 0.9893390191897654,
      I: 0.9876543209876543,
      E: 0.9850746268656716,
    },
    SECTION: {
      B: 0.9986136783733828,
      I: 0.9987291619944682,
      E: 0.9976894639556377,
    },
    PAGE: {
      B: 0.8409090909090908,
      I: 1.0,
      E: 0.8409090909090908,
      S: 0.9391363481808908,
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
