from sam_audio import SAMAudio, SAMAudioProcessor
import torchaudio
import torch
import os
from huggingface_hub import login
login()
import tqdm
import random
import numpy as np
import pandas as pd 


def new_sdr(references, estimates):
    """
    Compute the SDR for a song
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references), dim=(2, 3))
    den = torch.sum(torch.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores
    
def set_seed(seed=42):
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # OS environment (some libraries use this)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # CuDNN determinism
    # Warning: This can slow down performance slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def power(signal):
    return torch.mean(signal**2)

def conv_torch(srcs, rirs):
    #RIRs are minibatch, channels, samples
    #srcs are minibatch, source, channels, samples
    batch_size = srcs.shape[0]
    nsources = srcs.shape[1]
    out = []
    for i in range(batch_size):
        out_src = []
        for k in range(nsources):
            left = torchaudio.functional.fftconvolve(
                    srcs[i, k, 0, :],
                    rirs[i, 0, :], 'same')
            right = torchaudio.functional.fftconvolve(
                srcs[i, k, 1, :],
                rirs[i, 1, :], 'same')
            mix = torch.stack([left, right])
            mix *= torch.sqrt(power(srcs[i, k, :, :]) / power(mix)) #keep power the same as input audio    
            out_src.append(mix)
        out_src = torch.stack(out_src)
        out.append(out_src)
    out = torch.stack(out)
    return out

def gen_estimate(mix):
    mix = mix[0]

    inshape = mix.shape
    
    inlist = torch.split(mix, 1_000_000, dim=1)
    
    estimates = []
    for description_u in descriptions_l:# = 'vocals'
        if description_u == 'audience':
            query = 'crowd'
        else:
            query = description_u
        
        estimate = []
        
        for chunk in inlist:
            
            batch = processor(
                audios=[chunk],
                descriptions=[query],
                ).to("cuda")
            
            with torch.inference_mode():
                result = model.separate(batch, predict_spans=False, reranking_candidates=1)
                
                estimate.append(result.target[0].cpu()[:chunk.shape[1]])
        estimates.append(torch.cat(estimate).repeat(2,1))
    
    return torchaudio.functional.resample(torch.stack(estimates), 48000, fs_o)

set_seed(42)
music_test_set_path = '/media/diskA/enric/musdbmoises/test'
rir_test_set_path = '/media/diskA/enric/parirset/test'
model = SAMAudio.from_pretrained("facebook/sam-audio-large")

processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
model = model.eval().cuda()

songs = os.listdir(music_test_set_path)
songs.sort()
rirs_paths = os.listdir(rir_test_set_path)
rirs_paths.sort()

step = len(rirs_paths) / len(songs)
rirs_paths = [rirs_paths[int(i * step)] for i in range(len(songs))]
rirs_paths = rirs_paths[:len(songs)]

drrs_factors = [1.0, 0.5, 0.25, 0.01]

d_clean = []
d_noisy = []
d_rev = []
d_noisyrev = []

descriptions_l = ['drums', 'bass', 'other', 'vocals', 'audience']

for i, song in tqdm.tqdm(enumerate(songs)):
    drums_o, fs_o = torchaudio.load(os.path.join(os.path.join(music_test_set_path, song), 'drums.wav'))
    bass, _ = torchaudio.load(os.path.join(os.path.join(music_test_set_path, song), 'bass.wav'))
    vocals, _ = torchaudio.load(os.path.join(os.path.join(music_test_set_path, song), 'vocals.wav'))
    other, _ = torchaudio.load(os.path.join(os.path.join(music_test_set_path, song), 'other.wav'))
    audience, _ = torchaudio.load(os.path.join(os.path.join(music_test_set_path, song), 'audience.wav'))

    rir, fs_rir = torchaudio.load(os.path.join(os.path.join(rir_test_set_path, rirs_paths[i])))


    drums = torchaudio.functional.resample(drums_o, fs_o, 48000)
    bass = torchaudio.functional.resample(bass, fs_o, 48000)
    vocals = torchaudio.functional.resample(vocals, fs_o, 48000)
    other = torchaudio.functional.resample(other, fs_o, 48000)
    audience = torchaudio.functional.resample(audience, fs_o, 48000)
    rir = torchaudio.functional.resample(rir, fs_rir, 48000)

    # now decrease a bit reverberation
    midx = torch.argmax(torch.abs(rir), axis=1)
    l = rir[0, midx[0]].clone().detach()
    r = rir[1, midx[1]].clone().detach()
    rir *= drrs_factors[i % 4]
    rir[0, midx[0]] = l
    rir[1, midx[1]] = r

    sources = torch.stack((drums, bass, other, vocals, audience)).unsqueeze(0)

    rirs = rir.unsqueeze(0)
    rev_sources = conv_torch(sources, rirs)


    noisy_mix = sources.sum(dim=1)
    clean_mix = sources[:, 0:4].sum(dim=1)
    rev_mix = rev_sources[:, 0:4].sum(dim=1)
    noisyrev_mix = rev_sources.sum(dim=1)

    estimates_clean = gen_estimate(clean_mix)
    estimates_noisy = gen_estimate(noisy_mix)
    estimates_rev = gen_estimate(rev_mix)
    estimates_noisyrev = gen_estimate(noisyrev_mix)

    sources = torchaudio.functional.resample(sources[0], 48000, fs_o)
    rev_sources = torchaudio.functional.resample(rev_sources[0], 48000, fs_o)

    # ['drums', 'bass', 'other', 'vocals', 'audience']
    sdr_clean = new_sdr(sources.unsqueeze(0), estimates_clean.unsqueeze(0))
    sdr_noisy = new_sdr(sources.unsqueeze(0), estimates_noisy.unsqueeze(0))
    sdr_rev = new_sdr(rev_sources.unsqueeze(0), estimates_rev.unsqueeze(0))
    sdr_noisyrev = new_sdr(rev_sources.unsqueeze(0), estimates_noisyrev.unsqueeze(0))

    d_clean.append({"model_name": 'SAMAudio',
            "eval": 'clean',
            "song": song,
            "rir": None,
            "source": 'drums',
            "sdr": sdr_clean[0,0].item()})
    d_clean.append({"model_name": 'SAMAudio',
            "eval": 'clean',
            "song": song,
            "rir": None,
            "source": 'bass',
            "sdr": sdr_clean[0,1].item()})
    d_clean.append({"model_name": 'SAMAudio',
            "eval": 'clean',
            "song": song,
            "rir": None,
            "source": 'other',
            "sdr": sdr_clean[0,2].item()})
    d_clean.append({"model_name": 'SAMAudio',
            "eval": 'clean', 
            "song": song,
            "rir": None,
            "source": 'vocals',
            "sdr": sdr_clean[0,3].item()})
    
    d_noisy.append({"model_name": 'SAMAudio',
            "eval": 'noisy',
            "song": song,
            "rir": None,
            "source": 'drums',
            "sdr": sdr_noisy[0,0].item()})
    d_noisy.append({"model_name": 'SAMAudio',
            "eval": 'noisy',
            "song": song,
            "rir": None,
            "source": 'bass',
            "sdr": sdr_noisy[0,1].item()})
    d_noisy.append({"model_name": 'SAMAudio',
            "eval": 'noisy',
            "song": song,
            "rir": None,
            "source": 'other',
            "sdr": sdr_noisy[0,2].item()})
    d_noisy.append({"model_name": 'SAMAudio',
            "eval": 'noisy', 
            "song": song,
            "rir": None,
            "source": 'vocals',
            "sdr": sdr_noisy[0,3].item()})
    d_noisy.append({"model_name": 'SAMAudio',
            "eval": 'noisy', 
            "song": song,
            "rir": None,
            "source": 'audience',
            "sdr": sdr_noisy[0,4].item()})


    d_rev.append({"model_name": 'SAMAudio',
            "eval": 'rev',
            "song": song,
            "rir": rirs_paths[i],
            "source": 'drums',
            "sdr": sdr_rev[0,0].item()})
    d_rev.append({"model_name": 'SAMAudio',
            "eval": 'rev',
            "song": song,
            "rir": rirs_paths[i],
            "source": 'bass',
            "sdr": sdr_rev[0,1].item()})
    d_rev.append({"model_name": 'SAMAudio',
            "eval": 'rev',
            "song": song,
            "rir": rirs_paths[i],
            "source": 'other',
            "sdr": sdr_rev[0,2].item()})
    d_rev.append({"model_name": 'SAMAudio',
            "eval": 'rev', 
            "song": song,
            "rir": rirs_paths[i],
            "source": 'vocals',
            "sdr": sdr_rev[0,3].item()})

    d_noisyrev.append({"model_name": 'SAMAudio',
            "eval": 'noisyrev', 
            "song": song,
            "rir": rirs_paths[i],
            "source": 'drums',
            "sdr": sdr_noisyrev[0,0].item()})
    d_noisyrev.append({"model_name": 'SAMAudio',
            "eval": 'noisyrev', 
            "song": song,
            "rir": rirs_paths[i],
            "source": 'bass',
            "sdr": sdr_noisyrev[0,1].item()})
    d_noisyrev.append({"model_name": 'SAMAudio',
            "eval": 'noisyrev', 
            "song": song,
            "rir": rirs_paths[i],
            "source": 'other',
            "sdr": sdr_noisyrev[0,2].item()})
    d_noisyrev.append({"model_name": 'SAMAudio',
            "eval": 'noisyrev', 
            "song": song,
            "rir": rirs_paths[i],
            "source": 'vocals',
            "sdr": sdr_noisyrev[0,3].item()})
    d_noisyrev.append({"model_name": 'SAMAudio',
            "eval": 'noisyrev', 
            "song": song,
            "rir": rirs_paths[i],
            "source": 'audience',
            "sdr": sdr_noisyrev[0,4].item()})

    
    df_clean = pd.DataFrame.from_dict(d_clean)
    df_rev = pd.DataFrame.from_dict(d_rev)
    df_noisy = pd.DataFrame.from_dict(d_noisy)
    df_noisyrev = pd.DataFrame.from_dict(d_noisyrev)

    df_clean.to_excel(os.path.join('/media/diskA/enric/parirset_models/results_clean/SAMAudio_test_rank1.xlsx'), index=False)
    df_rev.to_excel(os.path.join('/media/diskA/enric/parirset_models/results_rev/SAMAudio.xlsx'), index=False)
    df_noisy.to_excel(os.path.join('/media/diskA/enric/parirset_models/results_noisy/SAMAudio.xlsx'), index=False)
    df_noisyrev.to_excel(os.path.join('/media/diskA/enric/parirset_models/results_noisyrev/SAMAudio.xlsx'), index=False)
