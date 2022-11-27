import os

import numpy as np
import torch
import torch.nn as nn
from mgca.utils.segmentation_loss import MixedLoss
from pytorch_lightning import LightningModule

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SSLSegmenter(LightningModule):
    def __init__(self,
                 seg_model: nn.Module,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-6,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['seg_model'])
        self.model = seg_model
        self.loss = MixedLoss(alpha=10)

        # for param in self.model.encoder.parameters():
        #     param.requires_grad = False

    def shared_step(self, batch, batch_idx, split):
        x, y = batch
        logit = self.model(x)
        logit = logit.squeeze(dim=1)
        loss = self.loss(logit, y)
        prob = torch.sigmoid(logit)
        dice = self.get_dice(prob, y)

        if batch_idx == 0:
            img = batch[0][0].cpu().numpy()
            mask = batch[1][0].cpu().numpy()
            mask = np.stack([mask, mask, mask])

            layered = 0.6 * mask + 0.4 * img
            img = img.transpose((1, 2, 0))
            mask = mask.transpose((1, 2, 0))
            layered = layered.transpose((1, 2, 0))

            # self.logger.experiment.log(
            #     {"input_image": [wandb.Image(img, caption="input_image")]}
            # )
            # self.logger.experiment.log(
            #     {"mask": [wandb.Image(mask, caption="mask")]})
            # self.logger.experiment.log(
            #     {"layered": [wandb.Image(layered, caption="layered")]}
            # )
            # self.logger.experiment.log(
            #     {"pred": [wandb.Image(prob[0], caption="pred")]})

        # log_iter_loss = True if split == 'train' else False
        self.log(
            f"{split}_loss",
            loss.item(),
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        return_dict = {"loss": loss, "dice": dice}
        return return_dict

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

    def shared_epoch_end(self, step_outputs, split):
        loss = [x["loss"].item() for x in step_outputs]
        dice = [x["dice"] for x in step_outputs]
        loss = np.array(loss).mean()
        dice = np.array(dice).mean()

        self.log(f"{split}_dice", dice, on_epoch=True,
                 logger=True, prog_bar=True)

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def get_dice(self, probability, truth, threshold=0.5):
        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert probability.shape == truth.shape

            p = (probability > threshold).float()
            t = (truth > 0.5).float()

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)

            dice_neg = (p_sum == 0).float()
            dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

            dice_neg = dice_neg[neg_index]
            dice_pos = dice_pos[pos_index]
            dice = torch.cat([dice_pos, dice_neg])

        return torch.mean(dice).detach().item()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay
        )

        return optimizer

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
