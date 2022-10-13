import datetime
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from mgca.datasets.classification_dataset import MIMICImageDataset
from mgca.datasets.data_module import DataModule
from mgca.datasets.transforms import DataTransforms
from mgca.models.backbones.encoder import ImageEncoder

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SLModule(LightningModule):
    ''' PyTorch Lightning implementation of supervised pre-training'''
    def __init__(self,
                 img_encoder: str = "vit_base",
                 hidden_mlp: int = 2048,
                 emb_dim: int = 128,
                 num_classes: int = 5,
                 softmax_temperature: float = 0.07,
                 momentum: float = 0.9,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-4,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.img_encoder = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.fc = nn.Linear(self.img_encoder.text_feat_dim, num_classes)

    def forward(self, imgs):
        img_feat, _ = self.img_encoder(imgs)
        logits = self.fc(img_feat)

        return logits

    def shared_step(self, batch):
        # images, labels, paths
        imgs, y, _ = batch
        logits = self(imgs)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y.float())

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, prog_bar=True,
                 on_epoch=False, sync_dist=True, batch_size=self.hparams.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, prog_bar=True,
                 on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=0.0,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--learning_rate", type=float, default=2.5e-5)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--seed", type=int, default=42)
        return parser

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


def cli_main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = SLModule.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50

    seed_everything(args.seed)

    # define datamodule
    datamodule = DataModule(MIMICImageDataset, None,
                            DataTransforms, 0.1,
                            args.batch_size, args.num_workers)

    model = SLModule(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/sl/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=5),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="sl", save_dir=logger_dir, name=extension)
    wandb_logger.watch(model, log="all")
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
