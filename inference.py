import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
import GPUtil

from model.generator import Generator
from utils.hparams import HParam, load_hparam_str

MAX_WAV_VALUE = 32768.0

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = devices


def main(args):
    save_path = os.path.join(os.path.join(args.input_folder.split("/LJ")[0],
                                          "Result", args.checkpoint_path.split("/")[5]), args.checkpoint_path.split("/")[-1].split('.')[0])
    if not os.path.exists(save_path): os.makedirs(save_path)
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=False)

    with torch.no_grad():
        for melpath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, '*.mel'))):
            mel = torch.load(melpath)
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.cuda()

            audio = model.inference(mel)
            audio = audio.cpu().detach().numpy()

            out_path = melpath.replace('.mel', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
            out_path = os.path.join(save_path, out_path.split('/')[-1])
            write(out_path, hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="/DataCommon2/ksoh/DeepLearning_Application/config/default.yaml",
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default="/DataCommon2/ksoh/DeepLearning_Application/chkpt/First_training/First_training_0375.pt",
                        help="path of checkpoint pt file for evaluation") #  TODO: you should change the model weights for the evaluation
    parser.add_argument('-i', '--input_folder', type=str, default="/DataCommon2/ksoh/DeepLearning_Application/LJSpeech-1.1/valid",
                        help="directory of mel-spectrograms to invert into raw audio.")
    args = parser.parse_args()

    main(args)
