import torch
import albumentations as A
import numpy as np
from config import Config
from torch.utils.data import DataLoader


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
                A.Cutout(max_h_size=5, max_w_size=16),
                A.CoarseDropout(max_holes=4),
                ], p=0.5),
    ])


class BirdDataset(torch.utils.data.Dataset):

    def __init__(self, df, sr=Config.SR, duration=Config.DURATION, augmentations=None, train=True):

        self.df = df
        self.sr = sr
        self.train = train
        self.duration = duration
        self.augmentations = augmentations
        if train:
            self.img_dir = Config.train_images
        else:
            self.img_dir = Config.valid_images

    def __len__(self):
        return len(self.df)

    @staticmethod
    def normalize(image):
        image = image / 255.0
        # image = torch.stack([image, image, image])
        return image

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        impath = self.img_dir + f"{row.filename}.npy"

        image = np.load(str(impath))[:Config.MAX_READ_SAMPLES]

        ########## RANDOM SAMPLING ################
        if self.train:
            image = image[np.random.choice(len(image))]
        else:
            image = image[0]

        #####################################################################

        image = torch.tensor(image).float()

        if self.augmentations:
            image = self.augmentations(image.unsqueeze(0)).squeeze()

        image.size()

        image = torch.stack([image, image, image])

        image = self.normalize(image)

        return image, torch.tensor(row[17:]).float()


def get_fold_dls(df_train, df_valid):

    ds_train = BirdDataset(
        df_train,
        sr=Config.SR,
        duration=Config.DURATION,
        augmentations=None,
        train=True
    )
    ds_val = BirdDataset(
        df_valid,
        sr=Config.SR,
        duration=Config.DURATION,
        augmentations=None,
        train=False
    )
    dl_train = DataLoader(
        ds_train, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=Config.batch_size, num_workers=2)
    return dl_train, dl_val, ds_train, ds_val
