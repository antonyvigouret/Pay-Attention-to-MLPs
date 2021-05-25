import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import configs
from input_pipeline import get_loaders
from models import VisiongMLP


logger = logging.getLogger(__name__)


def train(args, model, train_loader):
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    for epoch in range(args.num_train_epochs):
        loop = tqdm(train_loader)
        losses = []
        for images, labels in loop:
            images = images.to(args.device)
            preds = model(images)
            loss = criterion(preds, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            tr_loss += loss.item()

            optimizer.step()
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            losses.append(loss.item())
            loop.set_description(
                f"Epoch: {epoch+1}/{args.num_train_epochs} | Epoch loss: {sum(losses)/len(losses)}"
            )

    tb_writer.close()
    return global_step


def main():
    # TODO: logger, argparse config
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=False,
        help="Model type selected in the list: " + ", ",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model checkpoints will be saved",
    )

    # Other parameters
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate.")
    parser.add_argument(
        "--num_train_epochs", default=10, type=int, help="Total number of epochs for training."
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU when training."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available.")
    parser.add_argument("--logging_steps", default=100, type=int, help="Log every X updates steps.")

    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        raise NotImplementedError
    args.device = device

    # TODO load model with user command line selected config

    model = VisiongMLP(**configs["Ti"], prob_0_L=[1, 0.5]).to(args.device)
    train_loader, eval_loader, test_loader = get_loaders(args.per_gpu_train_batch_size, eval_split=0.15)
    print("*** Training ***")
    train(args, model, train_loader)


if __name__ == "__main__":
    main()
