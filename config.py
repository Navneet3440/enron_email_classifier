import transformers

DEVICE = "cuda"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 2
EPOCHS = 10
ACC_CUTOFF = 0.75
LEARNING_RATE = 1e-5
RETRAIN = False
NUMBER_OF_CLASS = 8
RETRAIN_MODEL_LOC = "./inputs/bert-base-multilingual-cased_128_ds2_ba_lr1e-05.bin"
TRAINING_MODE = "ba"
BERT_PATH = "bert-base-uncased"
DATASET_FILE_VERSION_COUNT = 1
MODEL_PATH = f"./inputs/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_VERSION_COUNT}_{TRAINING_MODE}_lr{str(LEARNING_RATE)}.bin"
MODEL_PATH_2 = f"./inputs/dump/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_VERSION_COUNT}_{TRAINING_MODE}_lr{str(LEARNING_RATE)}_"
TRAINING_FILE = (
    f"./final_data.csv"
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
