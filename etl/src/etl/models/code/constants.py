NER_LABEL_LIST = [
    "O",  # outside any entity
    "B-SECTION",  # begin a SECTION span
    "I-SECTION",  # inside a SECTION span
    "B-ARTICLE",  # begin an ARTICLE span
    "I-ARTICLE",  # inside an ARTICLE span
    "B-PAGE",  # begin a PAGE span
    "I-PAGE",  # inside a PAGE span
]
NER_CKPT_PATH = "/Users/nikitabogdanov/PycharmProjects/merger_agreements/scripts/models/best-epoch=04-val_loss=0.0008.ckpt"

CLASSIFIER_VOCAB_PATH = "classifier_vocab.pkl"
CLASSIFIER_LABEL2IDX_PATH = "classifier_label2idx.pkl"
CLASSIFIER_CKPT_PATH = "path/to/best-epoch=04-val_loss=0.0008.ckpt"
