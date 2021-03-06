import transformers

DEVICE = "cuda"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 3e-5
RETRAIN = False
NUMBER_OF_CLASS = 6
RETRAIN_MODEL_LOC = "./inputs/bert-base-cased_512_ds1_ba_lr3e-05.bin"
TRAINING_MODE = "ba"
BERT_PATH = "bert-base-uncased"
DATASET_FILE_VERSION_COUNT = 1
MODEL_PATH = f"./inputs/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_VERSION_COUNT}_{TRAINING_MODE}_lr{str(LEARNING_RATE)}_"
MODEL_PATH_2 = f"./inputs/dump/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_VERSION_COUNT}_{TRAINING_MODE}_lr{str(LEARNING_RATE)}_"
TRAINING_FILE = (
    f"./final_data_forward_removed.csv"
)
if "cased" in BERT_PATH.split("-"):
    TOKENIZER = transformers.BertTokenizer.from_pretrained(
        BERT_PATH, do_lower_case=False
    )
else:
    TOKENIZER = transformers.BertTokenizer.from_pretrained(
        BERT_PATH, do_lower_case=True
    )
if "large" in BERT_PATH.split("-"):
    LINEAR_INPUT_SIZE = 1024
else:
    LINEAR_INPUT_SIZE = 768
