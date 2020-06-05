import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from os.path import isfile
import ast
import librosa
import librosa.display
from sklearn.metrics import f1_score, accuracy_score
from time import time


def pca_plot_2D(tracks, features, genre1, genre2, save=False, filename=None):
    small_df = tracks['set', 'subset'] <= 'small'
    one = tracks['track', 'genre_top'] == genre1
    two = tracks['track', 'genre_top'] == genre2
    X = features.loc[small_df & (one | two), 'mfcc']
    X = PCA(2).fit_transform(X)
    y = tracks.loc[small_df & (one | two), ('track', 'genre_top')]
    y = LabelEncoder().fit_transform(y)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(X[:,0], X[:,1], c=y, cmap='winter', alpha = 0.3)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title(f'PCA 2 components on {genre1} and {genre2}')
    plt.tight_layout()
    if save:
        plt.savefig(filename)


def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all'),
                   ('track', 'genres_top')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)

        COLUMNS = [('track', 'license'), ('artist', 'bio'),
                   ('album', 'type'), ('album', 'information')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks
    

def get_track_ids(audio_dir):
    ids = []
    for _, dirnames, files in os.walk(audio_dir):
        if dirnames == []:
            ids.extend(int(file[:-4]) for file in files)
    return ids


def get_audio_path(audio_dir, track_id):
    id_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, id_str[:3], id_str + '.mp3')

def create_spectogram(track_id, audio_dir):
    filename = get_audio_path(audio_dir, track_id)
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T

def genre_specs(track_id, genre, AUDIO_DIR, save=False):
    filename = get_audio_path(AUDIO_DIR, track_id)
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect, y_axis='mel', fmax=8000, x_axis='time', cmap='winter')
    plt.colorbar(format='%+2.0f dB')
    plt.title(str(genre))
    if save:
        plt.savefig('../images/mel_spec_' + str(genre) + '.png')

def plot_spect(track_id, audio_dir, filename=None):
    spect = create_spectogram(track_id, audio_dir)
    fig, ax = plt.subplots(figsize=(10,4))
    librosa.display.specshow(spect.T, y_axis='mel', fmax=8000, x_axis='time', cmap='winter')
    plt.colorbar(format='%+2.0f dB')
    ax.set_title(f'Mel Spectogram for Track {track_id}')
    if filename:
        plt.savefig(filename)


def create_array(df, audio_dir, genre_dict):
    genres = []
    X_spect = np.empty((0, 640, 128))
    count = 0
    for index, row in df.iterrows():
        try:
            count += 1
            track_id = int(row['track_id'])
            genre = str(row[('track', 'genre_top')])
            spect = create_spectogram(track_id, audio_dir)
            spect = spect[:640, :]
            X_spect = np.append(X_spect, [spect], axis=0)
            genres.append(genre_dict[genre])
            if count % 100 == 0:
                print("Currently processing: ", count)
        except:
            print("Couldn't process: ", count)
            continue
    y_arr = np.array(genres)
    return X_spect, y_arr

def splitDataFrameIntoSmaller(df, chunkSize = 1600): 
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

    
def predict_labels(clf, features, target):
    y_pred = clf.predict(features)
    return f1_score(target, y_pred, average='micro', pos_label = 1), accuracy_score(target, y_pred)


def train_predict(model, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"Training a {model.__class__.__name__} using a training set size of {len(X_train)}. . .")
    model.fit(X_train, y_train)
    f1, acc = predict_labels(model, X_train, y_train)
    print(f"Training set F1 score:   {f1:.4f}       | Accuracy: {acc:.4f}.")
    f1, acc = predict_labels(model, X_val, y_val)
    print(f"Validation set F1 score: {f1:.4f}       | Accuracy: {acc:.4f}.")
    f1, acc = predict_labels(model, X_test, y_test)
    print(f"Test set F1 score:       {f1:.4f}       | Accuracy: {acc:.4f}.")

def pareto_plot(df, title, genre=None, filename=None):
    if genre:
        gen = df[df['True']==genre].copy()
        gen.set_index('Track_ID', inplace=True)
        gen.reset_index(inplace=True)
        x = [1,2,3,4,5,6,7,8]
        y1 = [gen[gen[col]==genre].shape[0] for col in gen.columns[2:]]
        y2 = (np.array(y1)/100).cumsum()
        f = [f'{x:,.0%}' for x in y2]
    else:
        df.set_index('Track_ID', inplace=True)
        df.reset_index(inplace=True)
        x = [1,2,3,4,5,6,7,8]
        y1 = [df[df[col]==df['True']].shape[0] for col in df.columns[2:]]
        y2 = (np.array(y1)/800).cumsum()
        f = [f'{x:,.2%}' for x in y2]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x, y1, color='blue', alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(x, y2, color='red', marker='o', linestyle='--')
    ax2.set_yticklabels([])
    for i in range(8):
        ax2.annotate(f[i], (x[i]+0.2, y2[i]-0.01), fontweight='heavy')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Number Correct')
    ax.set_title(f'Prediction Accuracy for {title}')
    plt.tight_layout()
    if filename:
        plt.savefig('../images/' + filename + '.png')