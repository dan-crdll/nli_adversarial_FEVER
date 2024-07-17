import torch
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def collate_fn(batch):
    labels_ids = ['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION']
    batch_sentences = []
    labels = []
    
    for b in batch:
        l, s = b['label'], b['sequence']
        labels.append(labels_ids.index(l))
        batch_sentences.append(s)
    tokens = tokenizer(batch_sentences, padding=True, return_tensors='pt')
    labels = torch.tensor(labels).long()
    return tokens.input_ids[:, :512], tokens.attention_mask[:, :512], labels