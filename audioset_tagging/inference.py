import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from utils.utilities import create_folder, get_filename
from pytorch.models import *
from pytorch.pytorch_utils import move_data_to_device
import utils.config as config

import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import sys

labels = ["Shout", "PoliceSiren", "Gunshot", "NoClasss"]


def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.

    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # # Print audio tagging top probabilities
    # for k in range(10):
    #     print(np.array(labels)[sorted_indexes[k]])
    #     # print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
    #     #     clipwise_output[sorted_indexes[k]]))
    idx = [0,1,2,3]
    for id in idx:
      print('{}: {:.3f}'.format(np.array(labels)[id], 
            clipwise_output[id]))
    


def plot_sound_event_detection_result(framewise_output, file):
    """Visualization of sound event detection result. 

    Args:
      framewise_output: (time_steps, classes_num)
    """
    out_fig_path = f'results/{file}_3.png'
    plt.clf()
    os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)

    classwise_output = np.max(framewise_output, axis=0) # (classes_num,)

    # idxes = np.argsort(classwise_output)[::-1]
    # idxes = idxes[0:5]
    # print(idxes)

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    lines = []
    # for idx in idxes:
    #     line, = plt.plot(framewise_output[:, idx], label=ix_to_lb[idx])
    #     lines.append(line)
    idx = [0]
    framewise = 0
    print(framewise_output.shape)
    for i in idx:
        framewise += framewise_output[:, i]
    # x_values = np.arange(len(framewise)) / 100
    # line, = plt.plot(x_values, framewise, label='Disparos de armas de fogo')
    hist_gun = framewise
    line, = plt.plot( framewise, label='Disparos de armas de fogo')
    lines.append(line)

    idx = [1]
    framewise = 0
    for i in idx:
        framewise += framewise_output[:, i]
    line, = plt.plot(framewise, label='Gritos')
    lines.append(line)
    hist_shouts = framewise
    idx = [2]
    framewise = 0
    for i in idx:
        framewise += framewise_output[:, i]
    line, = plt.plot( framewise, label='Sirene de polícia')
    lines.append(line)
    hist_police = framewise


    plt.legend(handles=lines)
    plt.xlabel('Audio Frames')
    plt.ylabel('Probability')
    plt.yscale('log')
    plt.savefig(out_fig_path)
    print('Save fig to {}'.format(out_fig_path))
    return hist_gun, hist_shouts, hist_police
    

def audio_tagging(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, model_type='Cnn14', checkpoint_path=None, device='cuda', cuda=True):
    """Inference audio tagging result of an audio clip.
    """
    device = torch.device('cuda') if cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(4):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels


def sound_event_detection(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, model_type='Cnn14', checkpoint_path=None, device='cuda', cuda=True, interpolate_mode='nearest'):
    """Inference sound event detection result of an audio clip.
    """

    device = torch.device('cuda') if cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_k = 4  # Show top results
    top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    
    """(time_steps, top_k)"""

    # Plot result    
    stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
        hop_length=hop_size, window='hann', center=True)
    frames_num = stft.shape[-1]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    epsilon = 1e-10
    axs[0].matshow(np.log(np.abs(stft) + epsilon), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save sound event detection visualization to {}'.format(fig_path))

    return framewise_output, labels


if __name__ == '__main__':
    """Example of using panns_inferece for audio tagging and sound evetn detection.
    """
    device = 'cuda' # 'cuda' | 'cpu'
    print(labels)
    # for i, label in enumerate(labels):
    #     print(i, label)
    audio_path = f"{sys.argv[1]}"
    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]  # (batch_size, segment_samples)

    print('------ Audio tagging ------')
    (clipwise_output, embedding) = audio_tagging(checkpoint_path="/home/luisa/Documents/identificacao-momentos-relevantes/audioset/0_iterations.pth",  device=device)
    """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""

    # print_audio_tagging_result(clipwise_output[0])

    print('------ Sound event detection ------')
    framewise_output, _ = sound_event_detection(
        checkpoint_path="/home/luisa/Documents/identificacao-momentos-relevantes/audioset/0_iterations.pth",
        device=device, 
        interpolate_mode='nearest', # 'nearest',
        model_type="Cnn14_DecisionLevelMax"
    )
    # print(framewise_output)
    # print(framewise_output.shape)
    """(batch_size, time_steps, classes_num)"""

    hist_guns, hist_shout, hist_siren = plot_sound_event_detection_result(framewise_output, audio_path.split('/')[-1].split('.')[0])
    # import json
    # hist_guns = [float(value) for value in hist_guns]
    # hist_shout = [float(value) for value in hist_shout]
    # hist_siren = [float(value) for value in hist_siren]

    # hist_of = [0.2 * hist_shout[i] + 0.2 * hist_siren[i] + 0.6 * hist_guns[i] for i in range(len(hist_guns))]

    # with open(f'results/{audio_path.split("/")[-1].split(".")[0]}.json', 'w') as f:
    #     json.dump({'hist_guns': hist_guns, 'hist_shout': hist_shout, 'hist_siren': hist_siren, 'hist_of': hist_of}, f)
