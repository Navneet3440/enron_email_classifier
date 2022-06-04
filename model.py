import config
import transformers
import torch.nn as nn


class BERT_CLASSIFIER(nn.Module):
    def __init__(self):
        super(BERT_CLASSIFIER, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.5)
        self.out = nn.Linear(config.LINEAR_INPUT_SIZE, config.NUMBER_OF_CLASS)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output