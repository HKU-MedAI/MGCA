from typing import List
import torch
import torch.nn as nn
from collections import OrderedDict
from mgca.utils.yolo_loss import YOLOLoss
from mgca.datasets.data_module import DataModule
from mgca.datasets.transforms import DataTransforms
from mgca.datasets.detection_dataset import RSNADetectionDataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from mgca.utils.detection_utils import non_max_suppression
from pytorch_lightning import LightningModule


class SSLDetector(LightningModule):
    def __init__(self,
                 img_encoder: nn.Module,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-6,
                 imsize: int = 224,
                 conf_thres: float = 0.5,
                 iou_thres: List = [0.4, 0.45, 0.5,
                                    0.55, 0.6, 0.65, 0.7, 0.75],
                 nms_thres: float = 0.5,
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['img_encoder'])
        self.model = ModelMain(img_encoder)
        self.yolo_losses = []
        for i in range(3):
            self.yolo_losses.append(YOLOLoss(self.model.anchors[i], self.model.classes,
                                             (imsize, imsize)))
        self.val_map = MeanAveragePrecision(
            iou_thresholds=self.hparams.iou_thres)
        self.test_map = MeanAveragePrecision(
            iou_thresholds=self.hparams.iou_thres)

    def shared_step(self, batch, batch_idx, split):
        outputs = self.model(batch["imgs"])
        losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
        losses = []
        for _ in range(len(losses_name)):
            losses.append([])
        for i in range(3):
            _loss_item = self.yolo_losses[i](outputs[i], batch["labels"])
            for j, l in enumerate(_loss_item):
                losses[j].append(l)
        losses = [sum(l) for l in losses]
        loss = losses[0]

        self.log(f"{split}_loss", loss, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        if split != "train":
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            output = non_max_suppression(output, self.model.classes,
                                         conf_thres=self.hparams.conf_thres,
                                         nms_thres=self.hparams.nms_thres)

            targets = batch["labels"].clone()
            # cxcywh -> xyxy
            h, w = batch["imgs"].shape[2:]
            targets[:, :, 1] = (batch["labels"][..., 1] -
                                batch["labels"][..., 3] / 2) * w
            targets[:, :, 2] = (batch["labels"][..., 2] -
                                batch["labels"][..., 4] / 2) * h
            targets[:, :, 3] = (batch["labels"][..., 1] +
                                batch["labels"][..., 3] / 2) * w
            targets[:, :, 4] = (batch["labels"][..., 2] +
                                batch["labels"][..., 4] / 2) * h

            sample_preds, sample_targets = [], []
            for i in range(targets.shape[0]):
                target = targets[i]
                out = output[i]
                if out is None:
                    continue
                filtered_target = target[target[:, 3] > 0]
                if filtered_target.shape[0] > 0:
                    sample_target = dict(
                        boxes=filtered_target[:, 1:],
                        labels=filtered_target[:, 0]
                    )
                    sample_targets.append(sample_target)

                    out = output[i]
                    sample_pred = dict(
                        boxes=out[:, :4],
                        scores=out[:, 4],
                        labels=out[:, 6]
                    )

                    sample_preds.append(sample_pred)

            if split == "val":
                self.val_map.update(sample_preds, sample_targets)
            elif split == "test":
                self.test_map.update(sample_preds, sample_targets)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

    def validation_epoch_end(self, validation_step_outputs):
        map = self.val_map.compute()["map"]
        self.log("val_mAP", map, prog_bar=True,
                 on_epoch=True, sync_dist=True)
        self.val_map.reset()

    def test_epoch_end(self, test_step_outputs):
        map = self.test_map.compute()["map"]
        self.log("test_mAP", map, prog_bar=True,
                 on_epoch=True, sync_dist=True)
        self.test_map.reset()

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


class ModelMain(nn.Module):
    def __init__(self, backbone, is_training=True):
        super(ModelMain, self).__init__()
        self.training = is_training
        self.backbone = backbone
        self.anchors = torch.tensor([
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]]
        ]) * 224 / 416
        self.classes = 1

        _out_filters = self.backbone.filters
        #  embedding0
        final_out_filter0 = len(self.anchors[0]) * (5 + self.classes)
        self.embedding0 = self._make_embedding(
            [512, 1024], _out_filters[-1], final_out_filter0)
        #  embedding1
        final_out_filter1 = len(self.anchors[1]) * (5 + self.classes)
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(
            scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding(
            [256, 512], _out_filters[-2] + 256, final_out_filter1)
        #  embedding2
        final_out_filter2 = len(self.anchors[2]) * (5 + self.classes)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(
            scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding(
            [128, 256], _out_filters[-3] + 128, final_out_filter2)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks,
             stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x)

        # x2: bz, 512, 28, 28
        # x1: bz, 1024, 14, 14
        # x0: bz, 2048, 7, 7
        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)

        # out0: bz, 18, 7, 7
        # out1: bz, 18, 14, 14
        # out2: bz, 18, 28, 28
        return out0, out1, out2


if __name__ == "__main__":
    model = ModelMain()

    datamodule = DataModule(RSNADetectionDataset, None, DataTransforms,
                            0.1, 32, 1, 224)

    for batch in datamodule.train_dataloader():
        break
