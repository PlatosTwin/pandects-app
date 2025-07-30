NER_CKPT_PATH = "/Users/nikitabogdanov/Downloads/dev-ner-model.ckpt"
NER_LABEL_LIST = [
    "O",  # outside any entity
    "B-SECTION",  # begin a SECTION span
    "I-SECTION",  # inside a SECTION span
    "B-ARTICLE",  # begin an ARTICLE span
    "I-ARTICLE",  # inside an ARTICLE span
    "B-PAGE",  # begin a PAGE span
    "I-PAGE",  # inside a PAGE span
]

CLASSIFIER_CKPT_PATH = "/Users/nikitabogdanov/Downloads/dev-classifier-model.ckpt"
CLASSIFIER_VOCAB_PATH = "classifier_vocab.pkl"
CLASSIFIER_LABEL2IDX_PATH = "classifier_label2idx.pkl"
