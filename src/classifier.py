from torch import nn
from torch import optim
from torch.nn import functional as F
import lightning as L
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.accuracy import Accuracy


class Classifier(L.LightningModule):
    def __init__(self, model, embed_dim=768, out_classes=3):
        super(Classifier, self).__init__()
        
        self.model = model

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, out_classes),
        ) 
        self.criterion = nn.CrossEntropyLoss()
        
        self.f1_score = F1Score(task='multiclass', num_classes=3)
        self.accuracy = Accuracy(task='multiclass', num_classes=3)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
        
    def forward(self, sequence, attention):
        embeddings = self.model(sequence, attention).last_hidden_state
        class_token = embeddings.permute(0, 2, 1)
        class_token = F.adaptive_avg_pool1d(class_token, 1).squeeze(-1)
        out = self.classifier(class_token)
        
        return out
    
    def training_step(self, batch, batch_idx):
        x, a, y = batch
        
        pred = self.forward(x, a)
        loss = self.criterion(pred, y)
        
        f1 = self.f1_score(pred.detach().cpu(), y.detach().cpu())
        accuracy = self.accuracy(pred.detach().cpu(), y.detach().cpu())
        
        self.log('Loss/Train', loss, prog_bar=True)
        self.log('F1/Train', f1)
        self.log('Accuracy/Train', accuracy)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, a, y = batch
        
        pred = self.forward(x, a)
        loss = self.criterion(pred, y)
        
        f1 = self.f1_score(pred.detach().cpu(), y.detach().cpu())
        accuracy = self.accuracy(pred.detach().cpu(), y.detach().cpu())
        
        self.log('Loss/Test', loss, prog_bar=True)
        self.log('F1/Test', f1)
        self.log('Accuracy/Test', accuracy)
        
        return loss