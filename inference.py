import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import sys
import os
import transformers
from tqdm import tqdm


BERT_PATH = 'bert-base-uncased'
LINEAR_INPUT = 768
MAX_LEN = 512
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
INFER_BATCH_SIZE = 8

class BERT_CLASSIFIER(nn.Module):
        def __init__(self):
            super(BERT_CLASSIFIER, self).__init__()
            self.bert = transformers.BertModel.from_pretrained(BERT_PATH,return_dict=False)
            self.bert_drop = nn.Dropout(0.3)
            self.out = nn.Linear(LINEAR_INPUT, 6)

        def forward(self, ids, mask, token_type_ids):
            _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
            bo = self.bert_drop(o2)
            output = self.out(bo)
            return output


class BERTDataset:
    def __init__(self, sent):
        self.sent = sent
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.sent)

    def __getitem__(self, item):
        sent = str(self.sent[item])
        sent = " ".join(sent.split())

        inputs = self.tokenizer.encode_plus(
            sent,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "sent": sent,
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

def infer_fn(data_loader, model, device):
    model.eval()
    fin_outputs = []
    fin_sent = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sent = d["sent"]
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
#             print(outputs.size())
            fin_outputs.extend(outputs.max(1, keepdim=False)[1].cpu().detach().numpy().tolist())
            fin_sent.extend(sent)
    return fin_sent,fin_outputs


def run():
    csv_file_loc = './test_bert_inference.csv'
    model_location = './bert-base-uncased_512_ds1_ba_lr3e-05_3_best_balance_accuracy_forward_removed.bin'


    if os.path.exists(csv_file_loc):

        if csv_file_loc.split('.')[-1] == 'csv':
            df_inference = pd.read_csv(csv_file_loc)
        elif csv_file_loc.split('.')[-1] == 'txt':
            with open(csv_file_loc) as fl_fix:
                fl_content = fl_fix.read()
                df_inference = pd.DataFrame(fl_content.split('\n')[:-1],columns=['email_text'])
        else:
            print(f"Cannot recognize file type of {csv_file_loc} as either csv or txt")

        try:
            inference_dataset = BERTDataset(sent=df_inference.email_text.values)
        except:
            inference_dataset = BERTDataset(sent=df_inference.email_text.values)

        inference_dataloader = torch.utils.data.DataLoader(
            inference_dataset, batch_size=INFER_BATCH_SIZE, num_workers=1
        )   
    else:
        print(f"CSV file at {csv_file_loc} dosen't exist")


    if os.path.exists(model_location):
        model = BERT_CLASSIFIER()
        if torch.cuda.is_available():
            DEVICE = 'cuda'
            model.load_state_dict(torch.load(model_location))
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            model.load_state_dict(torch.load(model_location,map_location=torch.device("cpu")))
        model.to(device)
        model.eval()
        sent_out , labels = infer_fn(inference_dataloader, model, device)
        df_out = pd.DataFrame({'sentence':sent_out,'model_category_label':labels})
        df_out.to_csv('./bert_inference.csv', index =False)
    else:
        print(f"Model path at {model_location} dosen't exist")

if __name__ == '__main__':
    run()
