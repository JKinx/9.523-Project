"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image
import math
from scipy.ndimage.filters import gaussian_filter

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, blur_radius, blur_every):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.blur_radius = blur_radius
        self.blur_every = 5
        # self.gaussian_filter = get_gaussian_kernel(kernel_size = 1, sigma = self.blur_radius)
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, False)
        # Define optimizer for the image
        optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                # if index == 27:
                #     print(layer.shape)
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            # loss = -torch.mean(self.conv_output[10:14,10:14])
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()

            if loss == 0:
                print("zero loss")
                self.processed_image = self.processed_image + preprocess_image(np.uint8(np.random.uniform(0, 20, (224, 224, 3))), False)
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)

            if self.blur_every is not 0 and self.blur_radius > 0:
                if i % self.blur_every == 0:
                    for channel in range(3):
                        cimg = gaussian_filter(self.created_image[:,:,channel], self.blur_radius)
                        self.created_image[:,:,channel] = cimg
            # Save image
            if i % 30 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)
import sys

if __name__ == '__main__':
    cnn_layer = int(sys.argv[1])
    filter_pos = int(sys.argv[2])
    blur_radius = float(sys.argv[3])
    blur_every = int(sys.argv[4])
    print("layer is {} and filter is {}".format(cnn_layer, filter_pos))
    print("blur radius is {} every {} iterations".format(blur_radius, blur_every))
    # Fully connected layer is not needed
    pretrained_model = models.vgg19(pretrained=True).features
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos, blur_radius, blur_every)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
