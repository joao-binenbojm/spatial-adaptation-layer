import torch
import random

import torch
import random

class RandomChannelBlobCorruption:
    def __init__(self, p=0.3, min_blob_size=1, max_blob_size=5):
        self.p = p
        self.min_blob_size = min_blob_size
        self.max_blob_size = max_blob_size

    def __call__(self, emg_img):
        """
        emg_img: Tensor of shape (T, 1, H, W)
        """
        if random.random() < self.p:
            T, _, H, W = emg_img.shape

            blob_h = random.randint(self.min_blob_size, min(self.max_blob_size, H))
            blob_w = random.randint(self.min_blob_size, min(self.max_blob_size, W))

            top = random.randint(0, H - blob_h)
            left = random.randint(0, W - blob_w)

            # Zero out the spatial blob across all time frames
            emg_img[:, 0, top:top+blob_h, left:left+blob_w] = 0

        return emg_img


# class RandomSingleChannelCorruption:
#     def __init__(self, min_noise=1.0, max_noise=5.0):
#         self.min_noise = min_noise
#         self.max_noise = max_noise

#     def __call__(self, emg_img):
#         """
#         emg_img: Tensor of shape (T, 1, H, W)
#         """
#         T, _, H, W = emg_img.shape
#         h = random.randint(0, H - 1)
#         w = random.randint(0, W - 1)

#         corruption_type = random.choice(['zero', 'additive_noise'])

#         if corruption_type == 'zero':
#             emg_img[:, 0, h, w] = 0
#         else:
#             noise = random.uniform(self.min_noise, self.max_noise)
#             noise = torch.rand(T) * noise  # Positive uniform noise
#             emg_img[:, 0, h, w] += noise

#         return emg_img

import torch
import random

# ...existing code...

class RandomChannelCorruption:
    def __init__(self, n_channels=1, min_noise=1.0, max_noise=5.0):
        """
        n_channels: Number of unique spatial channels (h, w) to corrupt per call.
        min_noise, max_noise: Range for additive noise.
        """
        self.n_channels = n_channels
        self.min_noise = min_noise
        self.max_noise = max_noise

    def __call__(self, emg_img):
        """
        emg_img: Tensor of shape (T, 1, H, W)
        """
        T, _, H, W = emg_img.shape
        total_channels = H * W
        n = min(self.n_channels, total_channels)

        # Generate all possible (h, w) pairs and sample without replacement
        all_indices = [(h, w) for h in range(H) for w in range(W)]
        selected_indices = random.sample(all_indices, n)

        for h, w in selected_indices:
            corruption_type = random.choice(['zero', 'additive_noise'])
            if corruption_type == 'zero':
                emg_img[:, 0, h, w] = 0
            else:
                noise = random.uniform(self.min_noise, self.max_noise)
                noise = torch.rand(T) * noise  # Positive uniform noise
                emg_img[:, 0, h, w] += noise

        return emg_img



# # Compose transforms, for example:
# from torchvision import transforms

# emg_transforms = transforms.Compose([
#     RandomChannelBlobCorruption(p=0.3, min_blob_size=1, max_blob_size=5),
#     RandomSingleChannelCorruption(max_amplitude_scale=5.0),
# ])
