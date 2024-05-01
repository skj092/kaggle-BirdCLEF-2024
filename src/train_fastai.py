from fastai.vision.all import *
from fastai.callback.wandb import *

import wandb

# wandb.login(key = wandb-key)
wandb.init(project="birdclef2024", name="fastai-resnet50")
path = Path("/teamspace/studios/this_studio/kaggle-BirdCLEF-2024/data")
Path.BASE_PATH = Path
path.ls()

df_train = pd.read_csv(path / "train.csv")
df_valid = pd.read_csv(path / "valid.csv")

labels = df_train["primary_label"].unique()
lbldict = {k: v for v, k in enumerate(labels)}

num_classes = len(df_train.primary_label.unique())
birds = list(df_train.primary_label.unique())

missing_birds = list(
    set(list(df_train.primary_label.unique())).difference(
        list(df_valid.primary_label.unique())
    )
)

df_valid[missing_birds] = 0
df_valid = df_valid[df_train.columns]  # Fix order

# Assuming 'train' and 'valid' directories under 'specs'
df_train['folder'] = 'train'
df_valid['folder'] = 'valid'

def load_image(row):
    impath = path / "specs" / row['folder'] / (row["filename"] + ".npy")
    image = np.load(impath)[:5]  # Assuming you want the first 5 time slices, adjust as needed
    # image = image[np.random.choice(image.shape[0])]  # Randomly choose one time slice
    image = image[0]
    # Normalize and convert to RGB by repeating the channels
    image = np.stack([image]*3, axis=-1)  # Repeat the channel 3 times for RGB

    return Image.fromarray(image)  # Normalize and convert to uint8

lbltfm = Pipeline([ColReader("primary_label"), Categorize(vocab=birds)])
tfms = [[load_image, PILImage.create], lbltfm]

train_ds = Datasets(df_train, tfms=tfms)
valid_ds = Datasets(df_valid, tfms=tfms)


dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=64, after_item=[ToTensor], after_batch=[IntToFloatTensor()]).cuda()

learn = vision_learner(dls, resnet50, metrics=accuracy, cbs=WandbCallback())
learn.fine_tune(5)