import os
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from wav import get_wav_datasets
from SCNet import SCNet
import argparse
import yaml
import tqdm
from utils import new_sdr
import augment
from torch.cuda.amp import GradScaler
from loss import spec_rmse_loss
from ml_collections import ConfigDict
import json
from datetime import datetime
import torchaudio
import itertools

def get_model(config):

    model = SCNet(**config.model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, total_params

def power(signal):
    return torch.mean(signal**2)
    
class ParirSetRIRS(Dataset,):
    def __init__(self, main_path, mode, set, sample_rate=44100):
        self.maxlen = 60000 #maximum IR lenght in samples
        self.main_path = main_path
        self.mode = mode

        if mode == 'b1':
            folders = ['b1_gp']
        elif mode == 'd1':
            folders = ['b1_gp', 'd1_original']
        elif mode == 'd2':
            folders = ['b1_gp', 'd1_original', 'd2_capsules']
        elif mode == 'd3':
            folders = ['b1_gp', 'd1_original', 'd2_capsules', 'd3_beamforming']
        elif mode == 'd4':
            folders = ['b1_gp', 'd1_original', 'd2_capsules', 'd3_beamforming', 'd4_permute']
        elif mode == 'ob1':
            folders = ['b1_gp']
        elif mode == 'od1':
            folders = ['d1_original']
        elif mode == 'od2':
            folders = ['d2_capsules']
        elif mode == 'od3':
            folders = ['d3_beamforming']
        elif mode == 'od4':
            folders = ['d4_permute']
        elif mode == 'test':
            folders = ['test']
        else:
            print('unspecified RIR dataset!')
        
        wav_files = []
        for folder in folders:
            files = os.listdir(os.path.join(main_path, folder))
            for f in files:
                if '.wav' in f:
                    wav_files.append(os.path.join(os.path.join(main_path, folder), f))

        wav_files.sort()
        if set == 'train':
            wav_files = wav_files[:int(0.8*len(wav_files))]
        elif set == 'valid':
            wav_files = wav_files[int(0.8*len(wav_files)):]  
        else:
            if mode != 'test':
                print('unspecified dataset set!')   

        self.wav_files = wav_files
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        path = self.wav_files[idx]
        waveform, sr = torchaudio.load(path)
        if self.sample_rate and sr != self.sample_rate:
            print('resampling RIR.')
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            sr = self.sample_rate
            
        zeros_to_pad = self.maxlen - waveform.shape[1]
        if zeros_to_pad > 0:
            return torch.hstack((waveform, torch.zeros((2, zeros_to_pad))))
        else:
            return waveform[:, :self.maxlen]


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

def augment_drr(rirs, drrs_factors):
    # store direct path indexes
    midx = torch.argmax(rirs, axis=2)
    l = []
    r = []
    # store original direct path levels
    for i in range(len(midx)):
        l.append(rirs[i, 0, midx[i,0]])
        r.append(rirs[i, 1, midx[i,1]])
    vals = torch.stack((torch.stack(l), torch.stack(r))).T

    for i in range(len(midx)):
        # downscale RIR randomly
        rirs[i] *= drrs_factors[i]
        # restore direct path level
        rirs[i, 0, midx[i,0]] = vals[i, 0]
        rirs[i, 1, midx[i,1]] = vals[i, 1]
    return rirs


def main():
    print('Starting training script...')

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='./result/', help="path to config file")
    parser.add_argument("--config_path", type=str, default='../conf/config.yaml', help="path to save checkpoint")
    parser.add_argument("--rir_mode", type=str, default='b1', help="which parirset rirs we use")

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.isfile(args.config_path):
        print(f"Error: config file {args.config_path} does not exist.")
        sys.exit(1)
        
    with open(args.config_path, 'r') as file:
            config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

    torch.manual_seed(config.seed)
    model, total_params = get_model(config)
  
    # torch also initialize cuda seed if available
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    if config.optim.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.optim.lr,
            betas=(config.optim.momentum, config.optim.beta2),
            weight_decay=config.optim.weight_decay)
    else:
        print('Unsupported optimizer type. Please use "adam".')

    train_set, valid_set = get_wav_datasets(config.data)

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.misc.num_workers, drop_last=True)

    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False,
        num_workers=config.misc.num_workers)


    rirset_train = ParirSetRIRS(config.data.rirs, args.rir_mode, set='train')
    rirset_valid = ParirSetRIRS(config.data.rirs, args.rir_mode, set='valid')

    if rirset_train.sample_rate != config.data.samplerate:
        print('RIR dataset sample rate does not match config sample rate!')

    rirloader_train = DataLoader(rirset_train, batch_size=config.batch_size, shuffle=True, drop_last=True)
    rirloader_valid = DataLoader(rirset_valid, batch_size=config.batch_size, shuffle=False, drop_last=True)


    loaders = {"train": train_loader, "valid": valid_loader, "rir_train": rirloader_train, "rir_valid": rirloader_valid}
    scaler = GradScaler()
    stft_config = {
            'n_fft': config.model.nfft,
            'hop_length': config.model.hop_size,
            'win_length': config.model.win_size,
            'center': True,
            'normalized': config.model.normalized
        }

    augments = [augment.Shift(shift=int(config.data.samplerate * config.data.shift),
                                same=config.augment.shift_same)]
    if config.augment.flip:
        augments += [augment.FlipChannels(), augment.FlipSign()]
    for aug in ['scale', 'remix']:
        kw = getattr(config.augment, aug)
        if kw.proba:
            augments.append(getattr(augment, aug.capitalize())(**kw))
    augmentx = torch.nn.Sequential(*augments)

    #folder = args.save_path
    checkpoint_file = Path(args.save_path) / 'model.th'
    optimizer_file = Path(args.save_path) / 'optimizer.th'
    checkpoint_log = Path(args.save_path) / 'log.json'
    #config_out = Path(args.save_path) / 'config_out.yaml'
    cdict = config.to_dict()

    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_log, "r") as f:
            d = json.load(f)
        best_nsdr = d['best_nsdr']
        best_epoch = d['best_epoch']
        oepoch = d['epoch'] + 1
        train_losses = d['train_losses']
        val_losses = d['val_losses']
        val_nsdrs = d['val_nsdrs']
        val_drums_sdrs = d['val_drums_sdrs']
        val_bass_sdrs = d['val_bass_sdrs']
        val_other_sdrs = d['val_other_sdrs']
        val_vocals_sdrs = d['val_vocals_sdrs']
        epoch_times = d['epoch_times']
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_file, map_location='cuda'))
            optimizer.load_state_dict(torch.load(optimizer_file, map_location='cuda'))

        else:
            model.load_state_dict(torch.load(checkpoint_file, map_location='cpu'))
            optimizer.load_state_dict(torch.load(optimizer_file, map_location='cpu'))  

    else:
        print("No checkpoint found, starting from scratch.")
        epoch_times = []
        best_nsdr = 0
        best_epoch = 0
        oepoch = 0  # Start from the first epoch
        train_losses = []
        val_losses = []
        val_nsdrs = []
        val_drums_sdrs = []
        val_bass_sdrs = []
        val_other_sdrs = []
        val_vocals_sdrs = []    


    for epoch in range(oepoch, config.epochs):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting epoch...")
        epoch_times.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        train_loss = []
        val_loss = []
        val_nsdr = []
        val_drums_sdr = []
        val_bass_sdr = []
        val_other_sdr = []
        val_vocals_sdr = []

        # Adjust LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.optim.lr * (config.optim.decay_rate**((epoch)//config.optim.decay_step))
        
        # Train one epoch
        loaders['train'].sampler.epoch = epoch  # Update sampler for shuffling
        loaders['valid'].sampler.epoch = epoch  # 

        print(f"Epoch {epoch + 1}/{config.epochs}")
        model.train()

        # keep memory usage under control during validation
        memthld = config.batch_size * (config.data.segment - 1) * config.data.samplerate

        # For every batch:

        for sources, rirs in tqdm.tqdm(zip(loaders['train'], itertools.cycle(loaders['rir_train']))):
            if torch.cuda.is_available():
                sources = sources.cuda()
                rirs = rirs.cuda()
            # only during training, augment DRR in RIRs (randomly reduce reverberation):
            drrs_factors = torch.rand(rirs.shape[0])
            rirs = augment_drr(rirs, drrs_factors)

            sources = augmentx(sources)
            sources = conv_torch(sources, rirs)
            mix = sources.sum(dim=1)

            estimate = model(mix)
            loss = spec_rmse_loss(estimate, sources, stft_config)

            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            grad_norm = 0
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm()**2
                    grads.append(p.grad.data)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss.append(loss.item())

        train_losses.append(sum(train_loss) / len(train_loss))
        model.eval()

        with torch.no_grad():
            # For every utterance:
            for sources, rirs in tqdm.tqdm(zip(loaders['valid'], itertools.cycle(loaders['rir_valid']))):
                if sources.shape[3] > memthld:
                    sources = sources[:, :, :, :memthld]
                if torch.cuda.is_available():
                    sources = sources.cuda()
                    rirs = rirs.cuda()
                # only during val, scale DRRs in RIRs always in the same way:
                rirs = augment_drr(rirs, torch.linspace(0,1, rirs.shape[0]))
                sources = conv_torch(sources, rirs)
                mix = sources[:, 0]
                sources = sources[:, 1:]
                estimate = model(mix) 
                loss = spec_rmse_loss(estimate, sources, stft_config)
                
                val_loss.append(loss.item())
                nsdrs = new_sdr(sources, estimate.detach()).mean(0)
                

                val_drums_sdr.append(nsdrs[0].item())
                val_bass_sdr.append(nsdrs[1].item())
                val_other_sdr.append(nsdrs[2].item())
                val_vocals_sdr.append(nsdrs[3].item())
                val_nsdr.append(nsdrs.mean().item())
            val_losses.append(sum(val_loss) / len(val_loss))
            val_nsdrs.append(sum(val_nsdr) / len(val_nsdr))
            val_drums_sdrs.append(sum(val_drums_sdr) / len(val_drums_sdr))
            val_bass_sdrs.append(sum(val_bass_sdr) / len(val_bass_sdr))
            val_other_sdrs.append(sum(val_other_sdr) / len(val_other_sdr))
            val_vocals_sdrs.append(sum(val_vocals_sdr) / len(val_vocals_sdr))
            

            if val_nsdrs[-1] > best_nsdr:
                best_nsdr = val_nsdrs[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_file)
                torch.save(optimizer.state_dict(), optimizer_file)
                print(f"Epoch: {epoch}. New best NSDR: {best_nsdr:.4f}. Saving at {checkpoint_file}.")
            # save log
            d = {
                    'epoch': epoch,
                    'best_nsdr': best_nsdr,
                    'best_epoch': best_epoch,
                    'val_nsdrs': val_nsdrs,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_drums_sdrs': val_drums_sdrs,
                    'val_bass_sdrs': val_bass_sdrs,
                    'val_other_sdrs': val_other_sdrs,
                    'val_vocals_sdrs': val_vocals_sdrs,
                    'total_params': total_params,
                    'epoch_times': epoch_times,
                    'config': cdict
                }
            with open(checkpoint_log, "w") as f:
                json.dump(d, f, indent=4)        
                    
if __name__ == "__main__":
    main()

