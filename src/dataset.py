from utils import compute_melspec, mono_to_color
import torch
import albumentations as A
import numpy as np
from config import Config
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
import librosa as lb


def get_train_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.Cutout(max_h_size=5, max_w_size=16),
                    A.CoarseDropout(max_holes=4),
                ],
                p=0.5,
            ),
        ]
    )


class BirdDataset(Dataset):

    def __init__(
        self, df, sr=Config.SR, duration=Config.DURATION, augmentations=None, train=True
    ):

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

        image = np.load(str(impath))[: Config.MAX_READ_SAMPLES]
        print(f"Image shape: {image.shape}")

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
        df_train, sr=Config.SR, duration=Config.DURATION, augmentations=None, train=True
    )
    ds_val = BirdDataset(
        df_valid,
        sr=Config.SR,
        duration=Config.DURATION,
        augmentations=None,
        train=False,
    )
    dl_train = DataLoader(
        ds_train, batch_size=Config.batch_size, shuffle=True, num_workers=2
    )
    dl_val = DataLoader(ds_val, batch_size=Config.batch_size, num_workers=2)
    return dl_train, dl_val, ds_train, ds_val


class BirdDatasetTest(Dataset):
    def __init__(
        self,
        data,
        sr=Config.SR,
        n_mels=128,
        fmin=0,
        fmax=None,
        duration=Config.DURATION,
        step=None,
        res_type="kaiser_fast",
        resample=True,
    ):

        self.data = data

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr // 2

        self.duration = duration
        self.audio_length = self.duration * self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def audio_to_image(self, audio):
        melspec = compute_melspec(audio, self.sr, self.n_mels, self.fmin, self.fmax)
        image = mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = []
        for i in range(self.audio_length, len(audio) + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            audios.append(audio[start:end])

        if len(audios[-1]) < self.audio_length:
            audios = audios[:-1]

        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        return images

    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "path"])
