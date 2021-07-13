import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm

from model import CNN, PL_CNN

parser = argparse.ArgumentParser()

# input arguments
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--batch_size', type=int, default=247)
parser.add_argument('--n_gpus', type=int, default=0)
parser.add_argument('--root', type=str, default='')
parser.add_argument('--save_path', type=str, default='./saved/pl/best_val_loss.ckpt')
parser.add_argument('--mode', type=str, choices=['pl', 'pytorch'], default='pl')

# save args
args = parser.parse_args()


# dataloader
def make_dataloader():
    test_dataset = datasets.MNIST(root=args.root, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return test_loader


# pytorch version
def go_pytorch():
    # device & seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # dataloader
    test_loader = make_dataloader()

    # model
    model = CNN().to(device)
    model.load_state_dict(torch.load(args.save_path, map_location=device))

    # testing
    with torch.no_grad():
        correct, total = 0, 0

        for x, y in tqdm(test_loader, desc="[Testing]"):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            correct_pred = torch.argmax(pred, 1) == y
            correct += correct_pred.float().sum()
            total += len(y)
        
        acc = correct / total

        print("Test Accuracy = {}".format(acc))


# pl version
def go_pl():
    # dataloader
    test_loader = make_dataloader()

    # model
    model = PL_CNN()
    model = model.load_from_checkpoint(args.save_path)

    # trainer
    trainer = pl.Trainer(gpus=args.n_gpus)
    trainer.test(model, test_loader)



def main():
    # pytorch version
    if args.mode == 'pl':
        go_pl()
    elif args.mode == 'pytorch':
        go_pytorch()

if __name__ == "__main__":
    main()