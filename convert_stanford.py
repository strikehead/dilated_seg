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
from utils import interp_map, pascal_palette


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

        npy_name = str.replace(path.basename(fname), '.regions.txt', '_label.npy')
        out_path = path.join(args.out_dir,npy_name)
        np.save(out_path, img_data)


        # #img = Image.fromarray(img_data)
        # img_name = str.replace(path.basename(fname), '.regions.txt', '_true.png')
        # #img.save(path.join(args.out_dir, img_name), 'png')
        # color_image = np.array(pascal_palette)[img_data.ravel()].reshape(
        #     img_data.shape + (3,))

        # out_path = path.join(args.out_dir,img_name)
        # print('Saving results to: ', out_path)
        # with open(out_path, 'wb') as out_file:
        #     Image.fromarray(color_image).save(out_file)

        #print(f'{f_cnt:05}/{len(files):05}')

if __name__ == '__main__':
    main()