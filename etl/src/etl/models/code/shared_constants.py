NER_CKPT_PATH = "/Users/nikitabogdanov/PycharmProjects/merger_agreements/appv2/etl/src/etl/models/model_files/dev-ner-model.ckpt"
NER_LABEL_LIST = [
    "O",  # outside any entity
    "B-SECTION",  # begin a SECTION span
    "I-SECTION",  # inside a SECTION span
    "B-ARTICLE",  # begin an ARTICLE span
    "I-ARTICLE",  # inside an ARTICLE span
    "B-PAGE",  # begin a PAGE span
    "I-PAGE",  # inside a PAGE span
]

CLASSIFIER_LABEL_LIST = ["front_matter", "toc", "body", "sig", "back_matter"]
CLASSIFIER_XGB_PATH = "/Users/nikitabogdanov/PycharmProjects/merger_agreements/appv2/etl/src/etl/models/model_files/xgb_multi_class.json"
CLASSIFIER_CKPT_PATH = "/Users/nikitabogdanov/PycharmProjects/merger_agreements/appv2/etl/src/etl/models/model_files/dev-classifier-model.ckpt"
