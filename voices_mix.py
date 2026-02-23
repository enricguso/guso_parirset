import numpy as np
from IPython.display import Audio 
import soundfile as sf
import os
import tqdm
import librosa as lsa
import pyrubberband as prb
import argparse 

def augment_voice(bvoc, fs, jitter):
    # bvoc is 1D array with the original transformed backing vocal
    # fs is sampling rate
    #jitter = 400
    
    # RANDOM AMPLITUDE ENVELOPE
    # Detectem els inicis de sons (onsets) que solen coincidir amb les síl·labes
    onset_frames = lsa.onset.onset_detect(y=bvoc, sr=fs, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
    onset_times = lsa.frames_to_time(onset_frames, sr=fs)
    onset_samps = lsa.frames_to_samples(onset_frames)

    #amp_env_val = (np.random.rand(len(onset_samps)) + 0.5)
    amp_env_val = 3 * np.random.rand(len(onset_samps))+0.3

    amp_env = list(np.linspace(1, amp_env_val[0], onset_samps[0])) # first value, from zero
    for i, env_val in enumerate(amp_env_val):
        if i<(len(amp_env_val)-1):
            amp_env += list(np.linspace(env_val, amp_env_val[i+1], onset_samps[i+1]-onset_samps[i]))
    amp_env += list(np.linspace(amp_env_val[-1], 1, len(bvoc)-onset_samps[-1]))
    amp_env = np.array(amp_env)
    out1 = bvoc * amp_env
    
    # TIME-STRETCHING
    time_map = []
    prev_offset = 0
    time_map.append((0, 0))
    for i, onset in enumerate(onset_samps):
        noise = np.random.randint(low=0, high=int(fs * jitter/1000))
        act_onset = onset + prev_offset
        act_onset += noise
        if i < len(onset_samps)-1:
            if act_onset > onset_samps[i+1] :
                prev_offset = act_onset - onset_samps[i+1]
            else:
                prev_offset = 0
        time_map.append((onset, act_onset))
    if act_onset < len(bvoc):
        time_map.append((len(bvoc), len(bvoc)))
    else:
        time_map.append((len(bvoc), act_onset))

    with open("time_map.txt", "w") as f:
        for a, b in time_map:
            f.write(f"{a}, {b}\n")
    stretched = prb.pyrb.timemap_stretch(out1, fs, time_map)

    if len(stretched) < len(bvoc):
        # pad with zeros
        padded = np.zeros(len(bvoc))
        padded[:len(stretched)] = stretched
        return padded
    return stretched[:len(bvoc)]

fs=44100

already = os.listdir('/media/diskA/enric/musdbmoises/mixed_voices/')

songs = os.listdir('/media/diskA/enric/musdbmoises/voices')
songs.sort()

blocks = 0 # number of blocks to split the processing, 4 means 5 blocks (0,1,2,3,4)
parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, choices=range(blocks+1), help='Integer from 0 to '+str(blocks))
args = parser.parse_args()

if args.n <   blocks :
    songs = songs[args.n * len(songs)//(blocks+1) : (args.n + 1) * len(songs)//(blocks+1)]
else:
    songs = songs[args.n * len(songs)//(blocks+1) : ]


for song in songs :
    if song.replace('.npy', '.wav') not in already :
        print('processing '+song)
        voices = np.load('/media/diskA/enric/musdbmoises/voices/'+song)
        augmented = np.zeros_like(voices)
        for i in tqdm.tqdm(range(len(voices))):
            augmented[i,:] = augment_voice(voices[i,:], fs, np.random.randint(low=300, high=500))
        out = np.zeros((2, augmented.shape[1]))
        for voice in augmented[:32]:
            rolled = np.roll(voice, np.random.randint(-12820, 12820))
            l = np.random.rand() * voice
            r = np.random.rand() * voice
            out += np.array([l, r])
        for voice in augmented[32:]:
            rolled = np.roll(voice, np.random.randint(-12820, 12820))
            l = np.random.rand() * voice
            r = np.random.rand() * voice
            out += 0.5*np.array([l, r])

        #normalize
        out /= np.max((np.max(out), np.abs(np.min(out))))

        out *= 0.99
        sf.write('/media/diskA/enric/musdbmoises/mixed_voices/'+song.replace('npy', 'wav'), out.T, fs)
        os.remove('/media/diskA/enric/musdbmoises/voices/' + song)