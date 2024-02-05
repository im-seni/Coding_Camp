import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T
from tqdm import tqdm

from dataset import FoodDataset
from model import vanillaCNN, vanillaCNN2, VGG19

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['CNN1', 'CNN2', 'VGG'], required=True, help='model architecture to train')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='the number of train epochs')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs('./save', exist_ok=True)
    os.makedirs(f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}', exist_ok=True)
    
    transforms = T.Compose([
        T.Resize((227,227), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomVerticalFlip(0.5),
        T.RandomHorizontalFlip(0.5),
    ])

    train_dataset = FoodDataset("./data", "train", transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataset = FoodDataset("./data", "val", transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    if args.model == 'CNN1':
        model = vanillaCNN()
    elif args.model == 'CNN2':
        model = vanillaCNN2()
    elif args.model == 'VGG': 
        model = VGG19()
    else:
        raise ValueError("model not supported")
    
    if not os.path.exists('./save'):
        os.makedirs('./save')
        
    ##########################   fill here   ###########################
        
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('(%(asctime)s) %(levelname)s: %(message)s', datefmt='%m/%d %I:%M:%S %p')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler) 

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Train Loop
    for ep in range(args.epoch): 
        model.train()
        size = len(train_loader)
        running_loss = 0.0

        logger.info(f"Training epoch {ep+1}")

        for batch, data in enumerate(tqdm(train_loader)):
            images = data['input'].to(device)
            labels = data['target'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            logger.debug(f"Step {batch} loss: {loss.item()}")

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {ep+1} loss: {avg_loss}")

        model.eval()
        correct = 0
        total = 0

        logger.info(f"Validating epoch {ep+1}")

        with torch.no_grad():
            for data in tqdm(val_loader):
                images = data['input'].to(device)
                labels = data['target'].to(device)
                

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)

                correct += (predicted == labels).sum().item()

        accuracy = correct / total  
     
        logger.info(f"Epoch {ep+1} accuracy = {accuracy}%")

        checkpoint_path = os.path.join(f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}', f'{ep}_score:{accuracy:.3f}.pth')
        torch.save(model.state_dict(), checkpoint_path) 