import torch
import torch.nn as nn
from tqdm import tqdm


def loss_fn(outputs, targets, weights):
    # print(f"output is ={outputs} \n with shape{outputs.shape}")
    # print(f"target is ={targets.viem(-1,1)} \n with shape{targets.view(-1,1).shape}")
    # print(outputs.shape,targets.shape,weights.shape,outputs.dtype,targets.dtype,weights.dtype)
    loss = nn.CrossEntropyLoss(weight=weights)(outputs, targets)
    return loss


def train_fn(data_loader, model, optimizer, device, scheduler, weights):
    model.train()
    epoch_train_loss = 0

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        weights = weights.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(outputs, targets, weights)
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return epoch_train_loss/len(data_loader)

def eval_fn(data_loader, model, device,weights):
    model.eval()
    fin_targets = []
    fin_outputs = []
    epoch_eval_loss = 0

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)
            weights = weights.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs,targets, weights)
            epoch_eval_loss += loss.item()
            # print(targets)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            # print(outputs)
            # print(outputs.max(1, keepdim=False)[1])
            fin_outputs.extend(outputs.max(1, keepdim=False)[1].cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets, epoch_eval_loss/len(data_loader)
