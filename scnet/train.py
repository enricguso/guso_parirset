import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
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

def get_model(config):

    model = SCNet(**config.model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, total_params


def main():
    print('Starting training script...')

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='./result/', help="path to config file")
    parser.add_argument("--config_path", type=str, default='../conf/config.yaml', help="path to save checkpoint")
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

    loaders = {"train": train_loader, "valid": valid_loader}
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
        for sources in tqdm.tqdm(loaders['train']):
            if torch.cuda.is_available():
                sources = sources.cuda()
            sources = augmentx(sources)
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
            for sources in tqdm.tqdm(loaders['valid']):
                if sources.shape[3] > memthld:
                    sources = sources[:, :, :, :memthld]
                if torch.cuda.is_available():
                    sources = sources.cuda()
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

            if val_nsdrs[-1] > best_nsdr:
                best_nsdr = val_nsdrs[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_file)
                torch.save(optimizer.state_dict(), optimizer_file)
                print(f"Epoch: {epoch}. New best NSDR: {best_nsdr:.4f}. Saving at {checkpoint_file}.")
            
if __name__ == "__main__":
    main()

