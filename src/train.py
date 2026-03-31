import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    preds, trues = [], []
    
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        pixel = batch['pixel_values'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits, _, _ = model(pixel, ids, mask)
        loss = model.compute_loss(logits, labels) if hasattr(model, 'compute_loss') else criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds.extend(logits.argmax(-1).detach().cpu().tolist())
        trues.extend(labels.detach().cpu().tolist())
    
    acc = accuracy_score(trues, preds)
    return total_loss / len(loader), acc


def setup_optimizer(model, learning_rate, weight_decay=1e-2):
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return optimizer


def setup_scheduler(optimizer, mode='min', factor=0.5, patience=1):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=mode, 
        factor=factor, 
        patience=patience
    )
    return scheduler
