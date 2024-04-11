import numpy as np
from utils import compute_melspec, mono_to_color
import librosa as lb
import soundfile as sf
import gc
import albumentations as A
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
import warnings
from models import BirdClefModel
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


class Config:
    num_classes = 264
    batch_size = 12
    PRECISION = 16
    seed = 2023
    model = "tf_efficientnet_b0_ns"
    pretrained = False
    use_mixup = False
    mixup_alpha = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = "/home/sonu/personal/BirdCLEF/data"
    train_images = data_root + "/specs/train/"
    valid_images = data_root + "/specs/valid/"
    train_path = data_root + "/train.csv"
    valid_path = data_root + "/valid.csv"

    test_path = data_root + "/test_soundscapes/"
    SR = 32000
    DURATION = 5
    LR = 5e-4

    model_ckpt = 'exp1/last.ckpt'


pl.seed_everything(Config.seed, workers=True)


df_train = pd.read_csv(Config.train_path)
Config.num_classes = len(df_train.primary_label.unique())

df_test = pd.DataFrame(
    [(path.stem, *path.stem.split("_"), path)
     for path in Path(Config.test_path).glob("*.ogg")],
    columns=["name", "id", "path"]
)
df_test["filename"] = df_test["name"]
print(df_test.shape)
df_test.head()


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
                A.Cutout(max_h_size=5, max_w_size=16),
                A.CoarseDropout(max_holes=4),
                ], p=0.5),
    ])


class BirdDataset(Dataset):
    def __init__(self, data, sr=Config.SR, n_mels=128, fmin=0, fmax=None, duration=Config.DURATION, step=None, res_type="kaiser_fast", resample=True):

        self.data = data

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
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
        melspec = compute_melspec(
            audio, self.sr, self.n_mels, self.fmin, self.fmax)
        image = mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr,
                                res_type=self.res_type)

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


ds_test = BirdDataset(
    df_test,
    sr=Config.SR,
    duration=Config.DURATION,
)


def predict(data_loader, model):

    model.to('cpu')
    model.eval()
    predictions = []
    for en in range(len(ds_test)):
        print(en)
        images = torch.from_numpy(ds_test[en])
        print(images.shape)
        with torch.no_grad():
            outputs = model(images).sigmoid().detach().cpu().numpy()
            print(outputs.shape)
#             pred_batch.extend(outputs.detach().cpu().numpy())
#         pred_batch = np.vstack(pred_batch)
        predictions.append(outputs)

    return predictions


print("Create Dataloader...")

ds_test = BirdDataset(
    df_test,
    sr=Config.SR,
    duration=Config.DURATION,
)


audio_model = BirdClefModel()

print("Model Creation")

model = BirdClefModel.load_from_checkpoint(
    Config.model_ckpt, train_dataloader=None, validation_dataloader=None)
print("Running Inference..")

preds = predict(ds_test, model)

gc.collect()
torch.cuda.empty_cache()
filenames = df_test.name.values.tolist()

bird_cols = list(pd.get_dummies(df_train['primary_label']).columns)
sub_df = pd.DataFrame(columns=['row_id']+bird_cols)

print(f"sub_df shape: {sub_df.shape}")

for i, file in enumerate(filenames):
    pred = preds[i]
    num_rows = len(pred)
    row_ids = [f'{file}_{(i+1)*5}' for i in range(num_rows)]
    df = pd.DataFrame(columns=['row_id']+bird_cols)

    df['row_id'] = row_ids
    df[bird_cols] = pred

    sub_df = pd.concat([sub_df, df]).reset_index(drop=True)
