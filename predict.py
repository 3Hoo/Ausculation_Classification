#!/usr/bin/python                                                       
# Author: Siddhartha Gairola (t-sigai at microsoft dot com))                 
                                                                    
import numpy as np
import os
import io
import math
import random
import pandas as pd

import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse
import cv2
from torch.utils import data
import cmapy

import nlpaug
import nlpaug.augmenter.audio as naa

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor

# load external modules
from utils import *
from image_dataloader import *
from nets.network_cnn import *

print ("Train import done successfully")

# input argmuments
parser = argparse.ArgumentParser(description='Lung Sound Classification')
parser.add_argument('--wav_file', default="", type=str, help='lung sound wav file')
parser.add_argument('--sample_rate', default=6000, type=int, help="wav file's sample rate")
parser.add_argument('--model_path', default="./model/cnn/ckpt_best_200_-1.pkl", type=str, help="saved model path")

args = parser.parse_args()

class image_loader2(Dataset):
    def __init__(self, data_dir, sample_rate = 6000, input_transform = None):

        # getting device-wise information
        self.file_to_device = {}
        device_id = 0

        #extracting the audiofilenames and the data for breathing cycle and it's label

        self.audio_data = []
        self.labels = "none"
        self.data_dir = os.path.dirname(data_dir)
        self.file_name = os.path.basename(data_dir)
        self.input_transform = input_transform

        mean, std = [0.5091, 0.1739, 0.4363], [0.2288, 0.1285, 0.0743]
        self.input_transform = Compose([ToTensor(), Normalize(mean, std)])

        # parameters for spectrograms
        self.sample_rate = sample_rate
        self.desired_length = 8
        self.n_mels = 64
        self.nfft = 256
        self.hop = self.nfft//2
        self.f_max = 2000
        self.train_flag = False

        self.dump_images = False
        self.filenames_with_labels = []

        print("Extracting Cycles...")
        self.cycle_list = []
        data = get_sound_samples_for_predict(self.file_name, self.data_dir, self.sample_rate)
        
        cycles_with_labels = [(d[0], d[3], self.file_name, cycle_idx, 0) for cycle_idx, d in enumerate(data[1:])]
        # cycles_with_labels = [ (잘린 오디오 파일1, 라벨1, filename, 0, 0), (잘린 오디오 파일2, 라벨2, filename, 1, 0) ]
        self.cycle_list.extend(cycles_with_labels)

        for idx, sample in enumerate(self.cycle_list) : 
            output = split_and_pad(sample, self.desired_length, self.sample_rate, types = 1)
            self.audio_data.extend(output)

    def __getitem__(self, index):

        audio = self.audio_data[index][0]
        
        aug_prob = random.random()
        if self.train_flag and aug_prob > 0.5:
            # apply augmentation to audio
            audio = gen_augmented(audio, self.sample_rate)

            # pad incase smaller than desired length
            audio = split_and_pad([audio, 0,0,0,0], self.desired_length, self.sample_rate, types=1)[0][0]
            
        # roll audio sample
        roll_prob = random.random()
        if self.train_flag and roll_prob > 0.5:
            audio = rollAudio(audio)
        
        # convert audio signal to spectrogram
        # spectrograms resized to 3x of original size
        audio_image = cv2.cvtColor(create_mel_raw(audio, self.sample_rate, f_max=self.f_max, 
            n_mels=self.n_mels, nfft=self.nfft, hop=self.hop, resz=3), cv2.COLOR_BGR2RGB)

        

        # blank region clipping
        audio_raw_gray = cv2.cvtColor(create_mel_raw(audio, self.sample_rate, f_max=self.f_max, 
            n_mels=self.n_mels, nfft=self.nfft, hop=self.hop), cv2.COLOR_BGR2GRAY)

        audio_raw_gray[audio_raw_gray < 10] = 0
        for row in range(audio_raw_gray.shape[0]):
            black_percent = len(np.where(audio_raw_gray[row,:]==0)[0])/len(audio_raw_gray[row,:])
            if black_percent < 0.80:
                break

        if (row+1)*3 < audio_image.shape[0]:
            audio_image = audio_image[(row+1)*3:, :, :]
        audio_image = cv2.resize(audio_image, (audio_image.shape[1], self.n_mels*3), interpolation=cv2.INTER_LINEAR)

        

        if self.dump_images:
            save_images((audio_image, self.audio_data[index][2], self.audio_data[index][3], 
                self.audio_data[index][5], self.audio_data[index][1]), self.train_flag)

        # label
        label = self.audio_data[index][1]

        # apply image transform 
        if self.input_transform is not None:
            audio_image = self.input_transform(audio_image)

        print("***********************")
        print(audio_image.shape)

        return audio_image, label

    def __len__(self):
        return len(self.audio_data)


def predict() :
    dataset = image_loader2(args.wav_file, args.sample_rate)

    net = model(num_classes=4).cuda()
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint)
    net.fine_tune(block_layer=5)
    print("Pre-trained Model Loaded:", args.model_path)

    net = nn.DataParallel(net, device_ids=[0])
    net.eval()

    data_loader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=False)

    for i, (image, label) in tqdm(enumerate(data_loader)): 
        image = image.cuda()
        output = net(image)
        score = output.cpu().detach().numpy()
        normal_score = score[0][0]
        rale_Score = score[0][1]
        rhonchi_score = score[0][2]
        wheezing_score = score[0][3]

        print()
        print(" [ SCORE ]")
        print("normal score : " + str(normal_score))
        print("rale score : " + str(rale_Score))
        print("rhonchi score : " + str(rhonchi_score))
        print("wheezing score : " + str(wheezing_score))
        print("-------------------------------")
        print()

        _, preds = torch.max(output, 1)

        print(" [ RESULT ]")
        if preds == 0 :
            print("normal")
        elif preds == 1 :
            print("rale")
        elif preds == 2 :
            print("rhonchi")
        elif preds == 3 : 
            print("wheezing")
        else : 
            print("error!")
        print()
        print()

if __name__ == "__main__":
    predict()
