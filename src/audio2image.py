from utils import compute_melspec, mono_to_color, crop_or_pad
from utils import get_audio_info
from sklearn.model_selection import train_test_split
import numpy as np
import librosa as lb
import soundfile as sf
import pandas as pd
from pathlib import Path
import joblib
from tqdm import tqdm
import re


def birds_stratified_split(df, target_col, test_size=0.2):
    class_counts = df[target_col].value_counts()
    # Birds with single counts
    low_count_classes = class_counts[class_counts < 2].index.tolist()

    df["train"] = df[target_col].isin(low_count_classes)

    train_df, val_df = train_test_split(
        df[~df["train"]],
        test_size=test_size,
        stratify=df[~df["train"]][target_col],
        random_state=42,
    )

    train_df = pd.concat([train_df, df[df["train"]]], axis=0).reset_index(drop=True)

    # Remove the 'valid' column
    train_df.drop("train", axis=1, inplace=True)
    val_df.drop("train", axis=1, inplace=True)

    return train_df, val_df


class Config:
    sampling_rate = 32000
    duration = 5
    fmin = 0
    fmax = None
    data_dir = Path("/home/sonu/personal/BirdCLEF/data")
    audios_path = data_dir / "train_audio"
    out_dir_train = Path("specs/train")
    out_dir_valid = Path("specs/valid")


def add_path_df(df):
    df["path"] = [str(Config.audios_path / filename) for filename in df.filename]
    df = df.reset_index(drop=True)
    pool = joblib.Parallel(2)
    mapper = joblib.delayed(get_audio_info)
    tasks = [mapper(filepath) for filepath in df.path]
    df2 = pd.DataFrame(pool(tqdm(tasks))).reset_index(drop=True)
    df = pd.concat([df, df2], axis=1).reset_index(drop=True)
    return df


class AudioToImage:
    def __init__(
        self,
        sr=Config.sampling_rate,
        n_mels=128,
        fmin=Config.fmin,
        fmax=Config.fmax,
        duration=Config.duration,
        step=None,
        res_type="kaiser_fast",
        resample=True,
        train=True,
    ):

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr // 2

        self.duration = duration
        self.audio_length = self.duration * self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample

        self.train = train

    def audio_to_image(self, audio):
        melspec = compute_melspec(audio, self.sr, self.n_mels, self.fmin, self.fmax)
        image = mono_to_color(melspec)
        #         compute_melspec(y, sr, n_mels, fmin, fmax)
        return image

    def __call__(self, row, save=False):

        audio, orig_sr = sf.read(row.path, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = [
            audio[i : i + self.audio_length]
            for i in range(0, max(1, len(audio) - self.audio_length + 1), self.step)
        ]
        audios[-1] = crop_or_pad(audios[-1], length=self.audio_length)
        print(f"Number of samples: {len(audios)}")
        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        if save:
            if self.train:
                path = Config.out_dir_train / f"{row.filename}.npy"
            else:
                path = Config.out_dir_valid / f"{row.filename}.npy"

            path.parent.mkdir(exist_ok=True, parents=True)
            np.save(str(path), images)
        else:
            return row.filename, images


def get_audios_as_images(df, train=True):
    pool = joblib.Parallel(1)
    converter = AudioToImage(
        step=int(Config.duration * 0.666 * Config.sampling_rate), train=train
    )
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for row in df.itertuples(False)]
    # take a subset of the data
    pool(tqdm(tasks[:5]))


def main():
    df = pd.read_csv(Config.data_dir / "train_metadata.csv")
    df["secondary_labels"] = df["secondary_labels"].apply(
        lambda x: re.findall(r"'(\w+)'", x)
    )
    df["len_sec_labels"] = df["secondary_labels"].map(len)
    print(df[df.len_sec_labels > 0].sample(3))
    train_df, valid_df = birds_stratified_split(df, "primary_label", 0.2)
    Config.out_dir_train.mkdir(exist_ok=True, parents=True)
    Config.out_dir_valid.mkdir(exist_ok=True, parents=True)
    train_df = add_path_df(train_df)
    valid_df = add_path_df(valid_df)

    get_audios_as_images(train_df, train=True)
    get_audios_as_images(valid_df, train=False)


if __name__ == "__main__":
    main()
