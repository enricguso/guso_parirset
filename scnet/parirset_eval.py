
import torch
from SCNet import SCNet
import tqdm
from utils import new_sdr
import os
import torchaudio
from IPython.display import Audio 
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # defaults
    parser.add_argument("--models_dir", type=str, default='/media/share/enric/scnet_models', help="main path to models checkpoints")
    parser.add_argument("--data_dir", type=str, default='/media/share', help="main path to datasets")
    parser.add_argument("--train_set", type=str, default='musdbmoises/train', help="dataset used for training")
    parser.add_argument("--out_path", type=str, default='/media/share/enric/scnet_models/results', help="path to save result")
    # compulsory
    parser.add_argument("--model_name", type=str, help="name of the model folder")
    parser.add_argument("--eval_set", type=str, help="test set folder")

    args = parser.parse_args()

    model_path = os.path.join(args.models_dir, args.model_name)

    test_set_path = os.path.join(args.data_dir, args.eval_set)

    model = SCNet()

    if torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(os.path.join(model_path,'model.th'), map_location='cuda'), strict=False)
    else:
        model.load_state_dict(torch.load(os.path.join(model_path,'model.th'), map_location='cpu'), strict=False)

    model.eval();

    songs = os.listdir(test_set_path)

    #while True:
    d = [] 
    for song in tqdm.tqdm(songs):
        mix, fs = torchaudio.load(os.path.join(os.path.join(test_set_path, song), 'mixture.wav'))
        drums, fs = torchaudio.load(os.path.join(os.path.join(test_set_path, song), 'drums.wav'))
        bass, fs = torchaudio.load(os.path.join(os.path.join(test_set_path, song), 'bass.wav'))
        vocals, fs = torchaudio.load(os.path.join(os.path.join(test_set_path, song), 'vocals.wav'))
        other, fs = torchaudio.load(os.path.join(os.path.join(test_set_path, song), 'other.wav'))
        if torch.cuda.is_available():
            mix = mix.cuda()
        mix = mix[:, :]
        with torch.no_grad():
            estimates = model(mix.unsqueeze(0)).squeeze()
            sdrs = new_sdr(torch.stack((drums, bass, other, vocals)).unsqueeze(0), estimates.cpu().unsqueeze(0))
            d.append({"model_name": args.model_name,
                    "train_set": args.train_set,
                    "eval_set": args.eval_set,
                    "song": song,
                    "source": 'drums',
                    "sdr": sdrs[0,0].item()})
            d.append({"model_name": args.model_name,
                    "train_set": args.train_set,
                    "eval_set": args.eval_set,
                    "song": song,
                    "source": 'bass',
                    "sdr": sdrs[0,1].item()})
            d.append({"model_name": args.model_name,
                    "train_set": args.train_set,
                    "eval_set": args.eval_set,
                    "song": song,
                    "source": 'other',
                    "sdr": sdrs[0,2].item()})
            d.append({"model_name": args.model_name,
                    "train_set": args.train_set,
                    "eval_set": args.eval_set,
                    "song": song,
                    "source": 'vocals',
                    "sdr": sdrs[0,3].item()})

    df = pd.DataFrame.from_dict(d)

    df.to_excel(os.path.join(args.out_path, args.model_name+'.xlsx'), index=False)
    print('Results saved to '+args.out_path+'/'+args.model_name+'.xlsx')