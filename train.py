from random import shuffle
import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from model import BERT_CLASSIFIER
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def calculate_weights(data_frame):
    weights = []
    print(data_frame.shape[0])
    for tag in sorted(data_frame['class_tag'].unique().tolist()):
        count = (data_frame['class_tag'] == tag).sum()
        weight_value = (1-(count/data_frame.shape[0]))/(config.NUMBER_OF_CLASS-1)
        print(f"{tag}-{weight_value}")
        weights.append(weight_value)
    print(f"sum of weights-{sum(weights)}")
    return weights


def run():
    dfx = pd.read_csv(config.TRAINING_FILE)
    dfx['class_tag'] = dfx['class_tag'] - 1
    print("Shape of datframe:",dfx.shape)
    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.2, random_state=42, stratify=dfx.class_tag.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    weights = calculate_weights(df_train)
    weights =torch.tensor(weights, dtype = torch.float)
    print("Shape of train datframe:",df_train.shape)
    print("Shape of validation dataframe:",df_valid.shape)

    train_dataset = dataset.BERTDataset(
        sent=df_train.email_text.values, target=df_train.class_tag.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=2, shuffle=True
    )

    valid_dataset = dataset.BERTDataset(
        sent=df_valid.email_text.values, target=df_valid.class_tag.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2, shuffle=True
    )

    # for bi, d in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
    #     ids = d["ids"]
    #     token_type_ids = d["token_type_ids"]
    #     mask = d["mask"]
    #     targets = d["targets"]
    #     print(targets)
    # return None
    device = torch.device(config.DEVICE)
    model = BERT_CLASSIFIER()
    if config.RETRAIN:
            DEVICE = 'cuda'
            model.load_state_dict(torch.load(config.RETRAIN_MODEL_LOC))
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.1,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    best_eval_loss = np.inf
    result = []

    for epoch in range(config.EPOCHS):
        epoch_train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, weights)
        outputs, targets, epoch_eval_loss = engine.eval_fn(valid_data_loader, model, device, weights)
        print(f" Model output is {outputs}")
        print(f"Actual target is {targets}")
        accuracy = metrics.accuracy_score(targets, outputs)
        print("Train loss = ", epoch_train_loss)
        print("Validation Loss = ", epoch_eval_loss)
        print("Accuracy Score =", accuracy)
        result.append((epoch+1,epoch_train_loss,epoch_eval_loss,accuracy))
        df_results = pd.DataFrame(result, columns = ['epoc','train_loss','valid_loss','valic_accuracy'])
        df_results.to_csv('./result.csv', index=False)

        if config.TRAINING_MODE == 'ba':
            best_eval_loss = np.inf
        if accuracy > best_accuracy and epoch_eval_loss < best_eval_loss:
            print("Saving Model state")
            torch.save(model.state_dict(), config.MODEL_PATH + f"{epoch}.bin")
            best_accuracy = accuracy
            best_eval_loss = epoch_eval_loss
        else:
            print("Saving model in dump folder")
            torch.save(model.state_dict(), config.MODEL_PATH_2 + f"{epoch}.bin")


if __name__ == "__main__":
    run()
