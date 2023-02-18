import numpy as np
import torch

def transform_image_to_patch(images, number_patches):
    n, c, h, w = images.shape
    assert h == w # Image must be square
    
    patches = torch.zeros(n, number_patches ** 2, h * w // number_patches ** 2)
    patch_size  = h  // number_patches
    
    for idx, image in enumerate(images):
        for i in range(number_patches):
            for j in range(number_patches):
                patch = image[:, i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
                patches[idx, i * number_patches + j] = patch.flatten()
                
    return patches

def calculate_positional_embeddings(sequence_length, d):
    # allows the model to understand where each patch would be placed in the original image
    # positional encoding adds low-frequency values to the first dimensions and higher-frequency values to the latter dimensions.
    positional_emb = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            if j%2 == 0:
                positional_emb[i, j] = np.cos(i / (10000 ** (2 * j / d)))
            else:
                positional_emb[i, j] = np.sin(i / (10000 ** (2 * j / d)))
    return positional_emb