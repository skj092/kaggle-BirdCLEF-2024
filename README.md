# BirdCLEF 2024

## To Identify Bird Species from their Sounds

## Provided Data

* **train_audios**: Contain 182 folders, each folder contain multiple audio files of particular bird species.
* **train_metadata.csv**: Contain metadata of each audio file.
* **unlabeled_soundscapes**: Contain multiple audio files of bird sounds.
* **eBIrd_Taxonomy_v2021.csv**: Data on the relationships between different species.
* **test_soundscapes**: Only the hidden rerun copy of the test soundscape dirctory will only be populated.
* **SampleSubmission.csv**: A sample submission file in the correct format. First column is the filename, and remaining are scores for each bird species.

## Evaluation Metric

* A version of macro-averaged ROC-AUC that skips classes which have no true positive labels.

## Dataset Info

* train audio are in different length and sample rate is 32000
* If an audio is for 6 second, when you read with soundfile, it will be 6*32000 = 192000 samples
* So the size of array will be (192000,)
* Since we have to predict for each 5 second, we will split the audio into 5 second.
* Now, will select one of the 5 second audio and convert it into melspectrogram of shape (3, 128, 201)
* When we make batch of it the shape will be (bs, 3, 128, 201)
* Model will give output of shape (bs, num_classes), i.e. (bs, 182)

## Pipeline

### 1. Convert the audio to spectrogram and train a simple image classification model

Pytorch Lightning Inference

1. import necessary libraries
2. Define configs
3. define functions:
    * computer melspec
    * mono_to_clolor
    * crop or pad
4. read train csv file
5. Update config num_classes wth totoal number of unique primary labels
6. Create test dataframe with filename, name, id ,path columns
7. def transform for augmentation
8. Define dataset class

## Convert Audio to Image

* read train metadata
* Create a new column secondary label and len_sec_labels
* Split the data into train ad valid
* Add path of corresponding audio file
* Call audion to image with duriation *samplerate* 0.666

* Read an audio file using soundfile.read method
* When you read an audio file using soundfile.read it return a tuple of length two.
* First value of the tuple is an array of length t * sr where `t` is the time of audio file and `sr` is the sample ratio of audio file. Second value of the tuple is an integer value which is sample ratio of the audio file.
* For example if I read an audio file of time 19 second, it will return a tuple (audio_vec, sr), where len(audio_vec) = 19* 32000. and sr = 32000.
* We chop this first array to multiple array considering each mini audio of length 5 second (5 * `sr`). So a 20s audio become 4 mini audios of 5 second.
* No, we convert this mini audio vector to spectrogram. We get spectrogram of shape (128 * x), we can confiture what shape we want.

## Submission
1. Data: 25% | Ecpoch 2 | LB: 0.57

## Resources

1. Convert Audio To Spectrogram: <https://www.kaggle.com/code/nischaydnk/split-creating-melspecs-stage-1>
2. Pytorch Lightning Inferece: <https://www.kaggle.com/code/nischaydnk/birdclef-2023-pytorch-lightning-inference>
3. Pytorch LIghtning training: <https://www.kaggle.com/code/nischaydnk/birdclef-2023-pytorch-lightning-training-w-cmap>
