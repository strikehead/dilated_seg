#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Segment images using weights from Fisher Yu (2016). Defaults to
settings for the Pascal VOC dataset.
'''

from __future__ import print_function, division

import argparse
import os

import numpy as np
from PIL import Image
from IPython import embed

from model import get_frontend, add_softmax, add_context
from utils import interp_map, pascal_palette

# Settings for the Pascal dataset
input_width, input_height = 900, 900
label_margin = 186

# Should be true whenever we are using pretrained weights as it is
has_context_module = True

def get_trained_model(args):
    """ Returns a model with loaded weights. """

    model = get_frontend(input_width, input_height)

    if has_context_module:
        model = add_context(model)

    model = add_softmax(model)

    def load_tf_weights():
        """ Load pretrained weights converted from Caffe to TF. """

        # 'latin1' enables loading .npy files created with python2
        weights_data = np.load(args.weights_path, encoding='latin1').item()

        for layer in model.layers:
            if layer.name in weights_data.keys():
                layer_weights = weights_data[layer.name]
                print (np.shape(layer_weights['weights']))
                print ('--' + np.shape(layer.get_weights()))
                layer.set_weights((layer_weights['weights'],
                                   layer_weights['biases']))

    def load_keras_weights():
        """ Load a Keras checkpoint. """
        model.load_weights(args.weights_path)

    if args.weights_path.endswith('.npy'): 
        load_tf_weights()
    elif args.weights_path.endswith('.hdf5'):
        load_keras_weights()
    else:
        raise Exception("Unknown weights format.")

    return model


def predict(args):
    ''' Runs a forward pass to segment the image. '''

    model = get_trained_model(args)

    img_basenames = [l.strip() for l in open(args.input_txt).readlines()]

    img_fnames = [os.path.join(args.input_dir, f) + '.jpg' for f in img_basenames]

    for i in range(len(img_fnames)):
        f = img_fnames[i]
        basename = img_basenames[i]
        # Load image and swap RGB -> BGR to match the trained weights
        image_rgb = np.array(Image.open(f)).astype(np.float32)
        image = image_rgb[:, :, ::-1] - args.mean
        image_size = image.shape

        # Network input shape (batch_size=1)
        net_in = np.zeros((1, input_height, input_width, 3), dtype=np.float32)

        output_height = input_height - 2 * label_margin
        output_width = input_width - 2 * label_margin

        # This simplified prediction code is correct only if the output
        # size is large enough to cover the input without tiling
        assert image_size[0] < output_height
        assert image_size[1] < output_width

        # Center pad the original image by label_margin.
        # This initial pad adds the context required for the prediction
        # according to the preprocessing during training.
        image = np.pad(image,
                       ((label_margin, label_margin),
                        (label_margin, label_margin),
                        (0, 0)), 'reflect')

        # Add the remaining margin to fill the network input width. This
        # time the image is aligned to the upper left corner though.
        margins_h = (0, input_height - image.shape[0])
        margins_w = (0, input_width - image.shape[1])
        image = np.pad(image,
                       (margins_h,
                        margins_w,
                        (0, 0)), 'reflect')

        # Run inference
        net_in[0] = image
        prob = model.predict(net_in)[0]

        # Reshape to 2d here since the networks outputs a flat array per channel
        prob_edge = np.sqrt(prob.shape[0]).astype(np.int)
        prob = prob.reshape((prob_edge, prob_edge, 9))

        # Upsample
        if args.zoom > 1:
            prob = interp_map(prob, args.zoom, image_size[1], image_size[0])

        # Recover the most likely prediction (actual segment class)
        prediction = np.argmax(prob, axis=2)

        pred_output_path = os.path.join(args.output_dir,'{}_pred.npy'.format(basename))
        np.save(pred_output_path,prediction)

        # Apply the color palette to the segmented image
        color_image = np.array(pascal_palette)[prediction.ravel()].reshape(
            prediction.shape + (3,))

        img_output_path = os.path.join(args.output_dir,'{}_seg.png'.format(basename))
        print('Saving results to: ', img_output_path)
        with open(img_output_path, 'wb') as out_file:
            Image.fromarray(color_image).save(out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', nargs='?', default='iccv09Data/val.txt',
                        help='txt input image')
    parser.add_argument('--input_dir', nargs='?', default='iccv09Data/images',
                        help='Required path to input images')
    parser.add_argument('--output_dir', default=None,
                        help='Path to segmented image')
    parser.add_argument('--mean', nargs='*', default=[102.93, 111.36, 116.52],
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    parser.add_argument('--zoom', default=8, type=int,
                        help='Upscaling factor')
    parser.add_argument('--weights_path', default='./dilation_pascal16.npy',
                        help='Weights file')

    args = parser.parse_args()

    # if not args.output_path:
    #     dir_name, file_name = os.path.split(args.input_path)
    #     args.output_path = os.path.join(      dir_name,   '{}_seg.png'.format( os.path.splitext(file_name)[0]))

    predict(args)


if __name__ == "__main__":
    main()
