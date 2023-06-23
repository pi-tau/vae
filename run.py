import os
from typing import Any

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as tt

from networks import Encoder, Decoder
from vae import VAE


class Trainee(pl.LightningModule):

    def __init__(self, model, config, log_dir, val_imgs):
        super().__init__()
        self.model = model
        self.config = config
        self.history = {
            "train_loss": [],
            "train_recon_loss": [],
            "train_kl_loss": [],
            "train_grad_norm": [],
            "val_loss": [],
            "learning_rate": [],
        }
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.val_imgs = val_imgs
        self.reconstruction_history = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), self.config["lr"], weight_decay=self.config["reg"],
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, total_iters=1000, # learning rate warm-up
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Compute the forward pass and the loss.
        x, y = batch
        total_loss, recon_loss, kl_loss = self.model.loss(x)
        self.history["train_loss"].append(total_loss.item())
        self.history["train_recon_loss"].append(recon_loss.item())
        self.history["train_kl_loss"].append(kl_loss.item())

        # Log the current learning rate.
        lr = self.lr_schedulers().get_last_lr()[0]
        self.history["learning_rate"].append(lr)

        return total_loss

    def on_before_optimizer_step(self, optimizer):
        total_grad_norm = torch.norm(torch.stack([
            torch.norm(p.grad) for p in self.model.parameters()
        ]))
        self.history["train_grad_norm"].append(total_grad_norm.item())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss, _, _ = self.model.loss(x)
        self.history["val_loss"].append(val_loss.item())

    def on_train_epoch_end(self):
        # Reconstruct the images from the validation set.
        imgs = self.model.reconstruct(self.val_imgs.to(self.device))
        imgs = (torch.clamp(imgs, min=-1, max=1) + 1.) / 2.
        self.reconstruction_history.append(imgs)

    def on_train_end(self):
        # Save the model.
        torch.save(self.model.cpu(), "vae.pt")

        # Plot the training losses.
        n_steps = len(self.history["train_loss"])
        n_epochs = len(self.history["val_loss"])
        xs = np.linspace(0, n_epochs, n_steps)
        fig, ax = plt.subplots()
        ax.set_title("Loss value during training")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.plot(xs, self.history["train_loss"], lw=0.6, label="Total Train Loss")
        ax.plot(xs, self.history["train_recon_loss"], lw=0.6, label="Reconstruction Loss")
        # ax.plot(xs, self.history["train_kl_loss"], lw=0.6, label="KL Loss")
        ax.plot(np.arange(n_epochs), self.history["val_loss"], lw=3., label="Validation Loss")
        ax.legend()
        fig.savefig(os.path.join(self.log_dir, "training_loss.png"))
        plt.close(fig)

        # Plot the gradient norm.
        fig, ax = plt.subplots()
        ax.set_title("Gradient Norm during training")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient Norm")
        ax.plot(self.history["train_grad_norm"], lw=0.6)
        fig.savefig(os.path.join(self.log_dir, "grad_norm.png"))
        plt.close(fig)

        # Plot the learning rate schedule.
        fig, ax = plt.subplots()
        ax.set_title("Learning rate schedule")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning rate")
        ax.plot(self.history["learning_rate"])
        fig.savefig(os.path.join(self.log_dir, "learning_rate.png"))
        plt.close(fig)

        # Plot the reconstructed images.
        val_imgs =  (self.val_imgs + 1.) / 2.
        n_epochs = len(self.reconstruction_history) + 1
        n_items = len(self.reconstruction_history[0])
        imgs = torch.stack([val_imgs] + self.reconstruction_history)
        imgs = imgs.transpose(0, 1).flatten(start_dim=0, end_dim=1)
        grid = torchvision.utils.make_grid(imgs, nrow=n_epochs)
        fig, ax = plt.subplots(figsize=(n_epochs, n_items), tight_layout={"pad":0})
        ax.axis("off")
        ax.imshow(grid.permute(1, 2, 0))
        fig.savefig(os.path.join(self.log_dir, "reconstruction_history.png"))
        plt.close(fig)


def load_CIFAR10(args):
    # The training and test sets will be scaled in the interval [-1, 1]. Note
    # that we are not normalizing the data with mean 0. and std 1., because
    # limiting the range would make predicting/reconstruction easier. In addition
    # the training set will be augmented with random crops. We will randomly
    # crop the image with the given scale and aspect ratio, and rescale the crop
    # afterwards to the original size.
    train_set = torchvision.datasets.CIFAR10("datasets", train=True, download=True)
    _, C, H, W = train_set.data.shape
    train_transform = tt.Compose([
        # tt.RandomResizedCrop(size=(H, W), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        tt.ToTensor(),
        tt.Normalize(mean=0.5, std=0.5),
    ])
    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(mean=0.5, std=0.5)])

    # Create train and test loaders with the defined transformations.
    train_set = torchvision.datasets.CIFAR10("datasets", train=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10("datasets", train=False, transform=test_transform)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return (train_loader, test_loader)


def main(args):
    pl.seed_everything(args.seed)
    plt.style.use("ggplot")

    # Load the CIFAR10 dataset.
    train_loader, test_loader = load_CIFAR10(args)

    # Initialize the variational autoencoder.
    vae = VAE(
        latent_dim=args.latent_dim,
        encoder=Encoder(in_chan=3, latent_dim=args.latent_dim),
        decoder=Decoder(out_chan=3, latent_dim=args.latent_dim),
    )

    def get_val_imgs():
        val_imgs, labels = [None] * 10, set()
        for x, y in test_loader:
            for x_, y_ in zip(x, y):
                y_ = y_.item()
                if y_ not in labels: val_imgs[y_] = x_; labels.add(y_)
                if len(labels) == 10: return torch.stack(val_imgs)

    # Train using Pytorch Lightning.
    trainee = Trainee(
        model=vae,
        config={"lr": args.lr, "reg": args.reg},
        log_dir="logs",
        val_imgs = get_val_imgs(),
    )
    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=True,
        max_epochs=args.epochs,
        enable_progress_bar=True,
        gradient_clip_val=args.clip_grad,
        # logger=None,
    )
    trainer.fit(
        model=trainee, train_dataloaders=train_loader, val_dataloaders=test_loader
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--reg", default=1e-4, type=float)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--latent_dim", default=16, type=int)
    parser.add_argument("--clip_grad", default=0, type=float)
    args = parser.parse_args()

    main(args)

#