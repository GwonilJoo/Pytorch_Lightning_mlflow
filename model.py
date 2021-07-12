import torch
from torch import nn, optim
import pytorch_lightning as pl
import mlflow

# pytorch version
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Linear(7*7*64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        
        out = out.view(out.size(0), -1) # out.view(-1, 7*7*64)
        out = self.fc(out)
        return out


# pytorch lightning version
class PL_CNN(pl.LightningModule):
    def __init__(self):
        super(PL_CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Linear(7*7*64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        
        out = out.view(out.size(0), -1) # out.view(-1, 7*7*64)
        out = self.fc(out)
        return out


    def loss_func(self, pred, y):
        creterion = nn.CrossEntropyLoss()
        return creterion(pred, y)
    
    
    def accuracy(self, pred, y):
        correct_pred = torch.argmax(pred, 1) == y
        return correct_pred.float().mean()
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_func(pred, y)

        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        return pred, y


    def validation_epoch_end(self, validation_step_outputs):
        avg_loss, correct, total = 0, 0, 0

        for (pred, y) in validation_step_outputs:
            avg_loss += self.loss_func(pred, y)
            correct += (torch.argmax(pred, 1) == y).float().sum()
            total += len(y)

        avg_loss /= len(validation_step_outputs)
        acc = correct / total

        metrics = {'val_acc': acc, 'val_loss': avg_loss}
        mlflow.log_metric('loss', float(avg_loss))
        mlflow.log_metric('acc', float(acc))
        self.log_dict(metrics)


    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        return pred, y

    def test_epoch_end(self, test_step_outputs):
        avg_loss, correct, total = 0, 0, 0

        for (pred, y) in test_step_outputs:
            avg_loss += self.loss_func(pred, y)
            correct += (torch.argmax(pred, 1) == y).float().sum()
            total += len(y)

        avg_loss /= len(test_step_outputs)
        acc = correct / total

        print(acc)

        metrics = {'test_acc': acc, 'test_loss': avg_loss}
        self.log_dict(metrics)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer