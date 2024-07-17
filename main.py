import sys 
sys.path.append('./')

import torch
import torch.nn.functional as F
import argparse
import seaborn as sns 
import matplotlib.pyplot as plt
import os 

from datasets import load_dataset
from src.classifier import Classifier
from src.collate_fn import collate_fn, tokenizer
from transformers import AutoModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.accuracy import Accuracy
from lightning.pytorch import seed_everything, Trainer

seed_everything(0)


def train(data, bert):
    train_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")['train']
    if data == 'adversarial':
        if 'augmented_train_set.pth' not in os.listdir('./'):
            print("Missing augmented training set")
            exit(0)
            
        adv_dataset = torch.load('./augmented_train_set.pth')
        
    val_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")['validation']
    
    train_dataset = construct_dataset(train_dataset)
    if data == 'adversarial':
        train_dataset.extend(construct_dataset(adv_dataset))
        
    val_dataset = construct_dataset(val_dataset)
    
    train_dl = DataLoader(train_dataset, 16, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_dataset, 16, collate_fn=collate_fn)
    
    model = Classifier(bert)
    if data == 'adversarial':
        model.load_state_dict(torch.load('./pretrained_model_original.pth'))
        
    model.train()
    
    trainer = Trainer(log_every_n_steps=1, precision='16-mixed', max_epochs=5)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    torch.save(model.state_dict(), f'./pretrained_model_{data}.pth')
    

def test(data, bert):
    if f'pretrained_model_{data}.pth' not in os.listdir('./'):
        print("Missing pretrained model")
        exit(0)
    
    if data == 'original':
        dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")['test']
    else:
        dataset = load_dataset("iperbole/adversarial_fever_nli")['test']
        
    test_dataset = construct_dataset(dataset)
        
    dataloader = DataLoader(test_dataset, 1, collate_fn=collate_fn)
    
    model = Classifier(bert)
    weights = torch.load(f'./pretrained_model_{data}.pth')
    
    model.load_state_dict(weights)
    model.to('cuda')
    model.eval()
    
    avg_loss = 0
    avg_acc = 0
    avg_f1 = 0

    cm = torch.zeros((3, 3))

    acc = Accuracy(task='multiclass', num_classes=3)
    f1 = F1Score(task='multiclass', num_classes=3)

    n = 0
    for b in tqdm(dataloader):
        x, a, y = b 
        
        with torch.no_grad():
            pred = model(x.to('cuda'), a.to('cuda'))
            loss = F.cross_entropy(pred.cpu(), y.cpu())
            
            avg_acc += acc(pred.cpu(), y.cpu())
            avg_f1 += f1(pred.cpu(), y.cpu())
            
            cm[pred[0].argmax(), y[0]] += 1
            
        avg_loss += loss
        n+= 1
        
    print(
        f"accuracy: {avg_acc / n} - f1: {avg_f1 / n} - loss: {avg_loss / n}"
    )
    sns.heatmap(cm / cm.sum(dim=0), cmap='Blues', annot=True)
    plt.show()
    

def construct_dataset(dataset):
    """
    Constucts an iterable dataset and preprocess the sentences.
    
    params:
        - dataset: a list of objects with fields: label, premise and hypothesis
    """
    ds = []
    for entry in tqdm(dataset):
        token = f"{entry['premise']} {tokenizer.sep_token} {entry['hypothesis']}"
        ds.append({
            'label': entry['label'],
            'sequence': token
        })
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--data", choices=["original", "adversarial"], required=True)
    
    args = parser.parse_args()
    
    mode = args.mode
    data = args.data
    
    bert = AutoModel.from_pretrained("distilbert/distilbert-base-uncased")
    
    if mode == 'train':
        train(data, bert)
    else:
        test(data, bert)

    