#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Converts the .txt image labels from the stanford background dataset to png mask images for training
according to train.py
'''

import argparse
import glob
import os
from os import path
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str, help='Input folder',
                        required=True)
    parser.add_argument('--out-dir', type=str, help='Output folder',
                        required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(path.join(args.in_dir, '*.txt')))

    assert len(files), 'no txt region files found in the input folder'

    try:
        os.makedirs(args.out_dir)
    except OSError:
        pass

    for f_cnt, fname in enumerate(files):
        img_data = []
        with open(fname,'rb') as f:
        	for line in f:
        		img_data.append(line.split())
        		# if img_data.size:
        		# 	img_data = img_data.vstack((img_data,line.split()))
        		# else:
        		# 	img_data = np.array(line.split())

        img_data = np.array(img_data, dtype=np.uint8)
        np.place(img_data,img_data==255,[8])
        print (img_data.shape)

        img = Image.fromarray(img_data)
        img_name = str.replace(path.basename(fname), '.regions.txt', '.png')
        img.save(path.join(args.out_dir, img_name), 'png')

        #print(f'{f_cnt:05}/{len(files):05}')

if __name__ == '__main__':
    main()