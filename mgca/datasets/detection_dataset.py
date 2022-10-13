import pickle
import random

import cv2
import numpy as np
from mgca.constants import *
from mgca.datasets.classification_dataset import BaseImageDataset
from mgca.datasets.transforms import *
from mgca.datasets.utils import read_from_dicom
from PIL import Image

random.seed(42)
np.random.seed(42)


class RSNADetectionDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224, max_objects=10):
        super().__init__(split, transform)
        if not os.path.exists(RSNA_DATA_DIR):
            raise RuntimeError(f"{RSNA_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(RSNA_DETECTION_TRAIN_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(RSNA_DETECTION_VALID_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "test":
            with open(RSNA_DETECTION_TEST_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        self.imsize = imsize

        self.filenames_list, self.bboxs_list = [], []
        for i in range(len(filenames)):
            bbox = bboxs[i]
            new_bbox = bbox.copy()
            new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.
            new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.
            new_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
            new_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
            n = new_bbox.shape[0]
            new_bbox = np.hstack([np.zeros((n, 1)), new_bbox])
            pad = np.zeros((max_objects - n, 5))
            new_bbox = np.vstack([new_bbox, pad])
            self.filenames_list.append(filenames[i])
            self.bboxs_list.append(new_bbox)

        self.filenames_list = np.array(self.filenames_list)
        self.bboxs_list = np.array(self.bboxs_list)
        n = len(self.filenames_list)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames_list = self.filenames_list[indices]
            self.bboxs_list = self.bboxs_list[indices]

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, index):
        filename = self.filenames_list[index]
        img_path = RSNA_IMG_DIR / filename
        x = read_from_dicom(
            img_path, None, None)

        x = cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2RGB)
        h, w, _ = x.shape
        x = cv2.resize(x, (self.imsize, self.imsize),
                       interpolation=cv2.INTER_LINEAR)
        x = Image.fromarray(x, "RGB")

        if self.transform:
            x = self.transform(x)

        y = self.bboxs_list[index]
        y[:, 1] /= w
        y[:, 3] /= w
        y[:, 2] /= h
        y[:, 4] /= h

        sample = {
            "imgs": x,
            "labels": y
        }

        return sample


class OBJCXRDetectionDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224, max_objects=20):
        # TODO: resize in detection is different from that in classification.
        super().__init__(split, transform)
        if not os.path.exists(OBJ_DATA_DIR):
            raise RuntimeError(f"{OBJ_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(OBJ_TRAIN_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(OBJ_VALID_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "test":
            with open(OBJ_TEST_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        self.filenames_list = []
        self.bboxs_list = []
        for i in range(len(bboxs)):
            filename = filenames[i]
            bbox = bboxs[i]
            bbox = np.array(bbox)

            padding = np.zeros((max_objects - len(bbox), 4))
            bbox = np.vstack((bbox, padding))
            bbox = np.hstack((np.zeros((max_objects, 1)), bbox))

            new_bbox = bbox.copy()
            # xminyminwh -> xywh
            new_bbox[:, 1] = bbox[:, 1] + bbox[:, 3] / 2
            new_bbox[:, 2] = bbox[:, 2] + bbox[:, 4] / 2

            self.filenames_list.append(filename)
            self.bboxs_list.append(new_bbox)

        n = len(self.filenames_list)
        self.filenames_list = np.array(self.filenames_list)
        self.bboxs_list = np.array(self.bboxs_list)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames_list = self.filenames_list[indices]
            self.bboxs_list = self.bboxs_list[indices]

        self.imsize = imsize

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, index):
        filename = self.filenames_list[index]
        if self.split == "train":
            img_path = OBJ_TRAIN_IMG_PATH / filename
        elif self.split == "valid":
            img_path = OBJ_VALID_IMG_PATH / filename
        elif self.split == "test":
            img_path = OBJ_TEST_IMG_PATH / filename
        else:
            raise RuntimeError()

        x = cv2.imread(str(img_path))
        h, w, _ = x.shape
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.imsize, self.imsize),
                       interpolation=cv2.INTER_LINEAR)

        if self.transform:
            x = self.transform(x)

        y = self.bboxs_list[index]
        y[:, 1] /= w
        y[:, 3] /= w
        y[:, 2] /= h
        y[:, 4] /= h

        sample = {
            "imgs": x,
            "labels": y
        }

        return sample


if __name__ == "__main__":
    dataset = RSNADetectionDataset(data_pct=0.01)
    print(len(dataset))
    print(dataset[0])
