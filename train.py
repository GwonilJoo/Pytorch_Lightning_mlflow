import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import nn, optim
import argparse
from tqdm import tqdm

from model import CNN, PL_CNN, PL_CNN2

import mlflow.pytorch
from utils import get_callback, print_auto_logged_info


parser = argparse.ArgumentParser()

# input arguments
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_gpus', type=int, default=0)
parser.add_argument('--root', type=str, default='')
parser.add_argument('--save_dir', type=str, default='saved/pl/')
parser.add_argument('--mode', type=str, choices=['pl', 'pytorch'], default='pl')

# save args
args = parser.parse_args()


# dataloader
def make_dataloader():
    dataset = datasets.MNIST(root=args.root, train=True, transform=transforms.ToTensor(), download=True)
    train_dataset, val_dataset = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_loader, valid_loader


# pytorch version
def go_pytorch():
    # device & seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # dataloader
    train_loader, valid_loader = make_dataloader()

    # model
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training
    best_val_loss, best_val_acc = 2000000000, 0

    
    for epoch in range(args.n_epochs):
        print('------------------')
        print(f'Epoch {epoch+1}')

        # train
        model.train()
        for x, y in tqdm(train_loader, desc=f'[Training]'):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # valid
        model.eval()
        with torch.no_grad():
            val_avg_loss, val_acc, total = 0, 0, 0
            
            for x, y in tqdm(valid_loader, desc=f'[Validation]'):
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                correct_pred = torch.argmax(pred, 1) == y

                val_acc += correct_pred.float().sum()
                total += len(y)

                loss = criterion(pred, y)

                val_avg_loss += loss / len(valid_loader)
        
            val_acc /= total

            if best_val_loss > val_avg_loss:
                best_val_loss = val_avg_loss
                PATH = args.save_dir + 'best_val_loss.pt'
                torch.save(model.state_dict(), PATH)
            
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                PATH = args.save_dir + 'best_val_acc.pt'
                torch.save(model.state_dict(), PATH)

            print("[Validation] loss = {}, acc = {}".format(val_avg_loss, val_acc))


# pl-mlflow version
def go_pl():
    pl.seed_everything(args.seed)

    # dataloader
    train_loader, valid_loader = make_dataloader()

    # model
    model = PL_CNN()

    # trainer
    
    trainer_args={
        'callbacks': get_callback(args.save_dir),
        'gpus': args.n_gpus,
        'max_epochs': args.n_epochs,
    }

    trainer = pl.Trainer(**trainer_args)

    mlflow.set_experiment("MNIST")
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        trainer.fit(model, train_loader, valid_loader)

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


def main():
    # pytorch version
    if args.mode == 'pl':
        go_pl()
    elif args.mode == 'pytorch':
        go_pytorch()
        #pass
        
        

if __name__ == "__main__":
    main()