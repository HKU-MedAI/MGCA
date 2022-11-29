import ipdb
import math
import numpy as np
import pandas as pd
from mgca.constants import *
import pickle
from shapely.geometry import LineString
from shapely.algorithms.polylabel import polylabel
from shapely.ops import unary_union
from sklearn.model_selection import train_test_split

np.random.seed(0)


OBJECT_SEP = ';'
ANNOTATION_SEP = ' '


def rectangle_box(anno):
    x = []
    y = []

    anno = anno[2:]
    anno = anno.split(ANNOTATION_SEP)
    for i in range(len(anno)):
        if i % 2 == 0:
            x.append(int(anno[i]))
        else:
            y.append(int(anno[i]))

    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    w = xmax - xmin
    h = ymax - ymin
    box = [xmin, ymin, w, h]
    return box


def polylabel_box(anno):
    polygon = anno.split(ANNOTATION_SEP)[1:]
    polygon = list(map(int, polygon))

    p = LineString(np.asarray(polygon + polygon[:2]).reshape(-1, 2))
    c = polylabel(p.buffer(100), tolerance=10)
    box = np.asarray(p.bounds).reshape(-1, 2)
    cxy = np.asarray(c)
    wh = np.abs(box - cxy)
    wh = np.maximum(wh[0], wh[1])
    box = [cxy - wh, wh * 2]  # xmin, ymin, w, h
    #unary_union([p, c, LineString(box)])
    return list(np.asarray(box).flat)


# source: https://github.com/xuyuan/xsd/blob/master/data/object_cxr_to_coco.ipynb
def annotation_to_bbox(annotation):
    bbox = []

    if not annotation:
        return bbox

    annotation_list = annotation.split(OBJECT_SEP)
    for anno in annotation_list:
        if anno[0] in ('0', '1'):
            box = rectangle_box(anno)
        elif anno[0] == '2':
            box = polylabel_box(anno)
        else:
            raise RuntimeError(anno[0])
        bbox.append(box)
    return bbox


def save_pkl(df, pkl_path):
    filenames, bboxs_list = [], []
    for row in df.itertuples():
        filenames.append(row.image_name)
        if row.annotation != row.annotation:
            bboxs_list.append(np.zeros((1, 4)))
        else:
            bboxs = annotation_to_bbox(row.annotation)
            bboxs_list.append(bboxs)

    filenames = np.array(filenames)
    bboxs_list = np.array(bboxs_list)
    with open(pkl_path, "wb") as f:
        pickle.dump([filenames, bboxs_list], f)


def main():
    ori_train_df = pd.read_csv(OBJ_ORIGINAL_TRAIN_CSV)
    # ori_train_df.dropna(subset=["annotation"], inplace=True)
    # ori_train_df.reset_index(drop=True, inplace=True)

    train_df, val_df = train_test_split(
        ori_train_df, test_size=0.1, random_state=0)

    save_pkl(train_df, OBJ_TRAIN_PKL)
    save_pkl(val_df, OBJ_VALID_PKL)

    test_df = pd.read_csv(OBJ_ORIGINAL_DEV_CSV)
    # test_df.dropna(subset=["annotation"], inplace=True)
    # test_df.reset_index(drop=True, inplace=True)
    save_pkl(test_df, OBJ_TEST_PKL)


if __name__ == "__main__":
    main()