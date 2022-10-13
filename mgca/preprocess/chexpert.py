import numpy as np
import pandas as pd
from mgca.constants import *

np.random.seed(0)


def preprocess_chexpert_5x200_data():
    df = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
    df = df.fillna(0)
    df = df[df["Frontal/Lateral"] == "Frontal"]

    task_dfs = []
    for i in range(len(CHEXPERT_COMPETITION_TASKS)):
        index = np.zeros(14)
        index[i] = 1
        df_task = df[
            (df["Atelectasis"] == index[0])
            & (df["Cardiomegaly"] == index[1])
            & (df["Consolidation"] == index[2])
            & (df["Edema"] == index[3])
            & (df["Pleural Effusion"] == index[4])
            & (df["Enlarged Cardiomediastinum"] == index[5])
            & (df["Lung Lesion"] == index[7])
            & (df["Lung Opacity"] == index[8])
            & (df["Pneumonia"] == index[9])
            & (df["Pneumothorax"] == index[10])
            & (df["Pleural Other"] == index[11])
            & (df["Fracture"] == index[12])
            & (df["Support Devices"] == index[13])
        ]
        df_task = df_task.sample(n=200, random_state=0)
        task_dfs.append(df_task)
    df_200 = pd.concat(task_dfs)

    return df_200


def preprocess_chexpert_data():
    try:
        df = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the chexpert dataset is \
            stored at {CHEXPERT_DATA_DIR}"
        )

    df_200 = preprocess_chexpert_5x200_data()
    df = df[~df[CHEXPERT_PATH_COL].isin(df_200[CHEXPERT_PATH_COL])]
    valid_ids = np.random.choice(
        len(df), size=CHEXPERT_VALID_NUM, replace=False)
    valid_df = df.iloc[valid_ids]
    train_df = df.drop(valid_ids, errors="ignore")

    print(f"Number of train samples: {len(train_df)}")
    print(f"Number of valid samples: {len(valid_df)}")
    print(f"Number of chexpert5x200 samples: {len(df_200)}")

    train_df.to_csv(CHEXPERT_TRAIN_CSV)
    valid_df.to_csv(CHEXPERT_VALID_CSV)
    df_200.to_csv(CHEXPERT_5x200)


if __name__ == "__main__":
    preprocess_chexpert_data()