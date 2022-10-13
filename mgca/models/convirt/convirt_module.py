import datetime
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from mgca.datasets.data_module import DataModule
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                             multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms
from mgca.models.backbones.encoder import BertEncoder, ImageEncoder
from torch import nn

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ConVIRT(LightningModule):
    ''' PyTorch Lightning implementation of ConVIRT
        https://arxiv.org/pdf/2010.00747.pdf
    '''

    def __init__(self,
                 img_encoder: str = "resnet_50",
                 hidden_mlp: int = 2048,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 freeze_bert: bool = False,
                 momentum: float = 0.9,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 1e-4,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.img_encoder = img_encoder
        self.freeze_bert = freeze_bert
        self.init_encoder()

    def init_encoder(self):
        self.img_encoder = ImageEncoder(
            model_name=self.img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=self.freeze_bert)

    def forward(self, batch):
        img_feat, _ = self.img_encoder(batch["imgs"])
        img_emb = self.img_encoder.global_embed(img_feat)
        img_emb = F.normalize(img_emb, dim=1)

        sent_feat, _, _, _ = self.text_encoder(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        sent_emb = self.text_encoder.global_embed(sent_feat)
        sent_emb = F.normalize(sent_emb, dim=1)

        return img_emb, sent_emb

    def info_nce_loss(self, out_1, out_2, temperature):
        bz = out_1.size(0)
        labels = torch.arange(bz).type_as(out_1).long()

        scores = out_1.mm(out_2.t())
        scores /= temperature
        scores1 = scores.transpose(0, 1)
        loss0 = nn.CrossEntropyLoss()(scores, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)

        return loss0 + loss1

    def shared_step(self, batch):
        img_emb, sent_emb = self(batch)
        loss = self.info_nce_loss(
            img_emb, sent_emb, self.hparams.softmax_temperature)

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
            min_lr=1e-8,
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
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--num_negatives", type=int, default=65536)
        parser.add_argument("--encoder_momentum", type=float, default=0.999)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=72)
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
    parser = ConVIRT.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50

    seed_everything(args.seed)

    # define datamodule
    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            DataTransforms, 1.,
                            args.batch_size, args.num_workers)

    model = ConVIRT(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/ConVIRT/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=5),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=5, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="ConVIRT", save_dir=logger_dir, name=extension)
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
