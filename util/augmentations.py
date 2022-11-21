import sys

import torch
import torch.fft as fft

import numpy as np

from typing import Any


class Rescaling(object):
    """
        Randomly rescale features of the sample.
    """
    def __init__(self, sigma=0.1) -> None:
        self.sigma = sigma

    def __call__(self, sample) -> Any:
        return sample * torch.normal(mean=torch.Tensor([1]), std=torch.Tensor([self.sigma]))

class Permutation(object):
    """
        Randomly permute features of the sample.
    """
    def __call__(self, sample) -> Any:
        return sample[..., torch.randperm(n=sample.shape[-2]), :]

class Jitter(object):
    """
        Add gaussian noise to the sample.
    """
    def __init__(self, sigma=0.03) -> None:
        self.sigma = sigma

    def __call__(self, sample) -> Any:
        return sample + torch.normal(mean=0, std=self.sigma, size=sample.shape)

class Shift(object):
    """
        Randomly shift the signal in the time domain.
    """
    def __init__(self, fs=250, padding_len_sec=30) -> None:
        self.padding_len = fs * padding_len_sec # padding len in ticks 

    def __call__(self, sample) -> Any:
        # define padding size 
        left_pad = int(torch.rand(1) * self.padding_len)
        right_pad = self.padding_len - left_pad

        # zero-pad the sample
        # note: the signal length is now extended by self.padding_len
        padded_sample = torch.nn.functional.pad(sample, (left_pad, right_pad), value=0)

        # get back to the original signal length
        if torch.rand(1) < 0.5:
            return padded_sample[..., :sample.shape[-1]]
        else:
            return padded_sample[..., right_pad:sample.shape[-1]+right_pad]

class TimeToFourier(object):
    """
        Go from time domain to frequency domain.
    """
    def __init__(self, factor=1) -> None:
        super().__init__()
        self.factor = factor

    def __call__(self, sample) -> torch.Tensor:
        sample_dims = sample.dim()

        # define the output length of the Fourier transform
        N = self.factor * sample.shape[-1] 
        
        # perform the Fourier transform and reorder the output to have negative frequencies first
        # note: the output of the Fourier transform is complex (real + imaginary part)
        X_f = 1/N * fft.fftshift(fft.fft(sample, n=N))

        # concatenate the real and imaginary parts 
        # note: the Fourier transform of a signal with only real parts is symmetric 
        #       thus only half of the transform is returned to save memory
        X_f_complex = torch.Tensor()

        if sample_dims == 2:
            for ch in range(X_f.shape[0]):
                real_part = torch.real(X_f[ch, int(N/2):]).unsqueeze(dim=0)
                imag_part = torch.imag(X_f[ch, int(N/2):]).unsqueeze(dim=0)

                complex_pair = torch.cat((real_part, imag_part), dim=0)
                X_f_complex = torch.cat((X_f_complex, complex_pair), dim=0)

        elif sample_dims == 3:
            X_f_bin_complex = torch.Tensor()
            
            for bin in range(X_f.shape[0]):
                for ch in range(X_f.shape[1]):
                    real_part = torch.real(X_f[bin, ch, int(N/2):]).unsqueeze(dim=0)
                    imag_part = torch.imag(X_f[bin, ch, int(N/2):]).unsqueeze(dim=0)

                    complex_pair = torch.cat((real_part, imag_part), dim=0)
                    X_f_bin_complex = torch.cat((X_f_bin_complex, complex_pair), dim=0)

                X_f_complex = torch.cat((X_f_complex, X_f_bin_complex.unsqueeze(dim=0)), dim=0)

        return X_f_complex

class FourierToTime(object):
    """
        Go from frequency domain to time domain.
    """
    def __init__(self, factor=1) -> None:
        super().__init__()
        self.factor = factor

    def __call__(self, sample) -> torch.Tensor:
        # define the output length of the Fourier transform
        N = self.factor * sample.shape[-1]
        
        # reorder the input to have positive frequencies first and perform the inverse Fourier transform
        # note: the output of the inverse Fourier transform is complex (real + imaginary part)
        x_t = N * fft.ifft(fft.ifftshift(sample), n=N)

        # return the real part
        return torch.real(x_t)

class CropResizing(object):
    """
        Randomly crop the sample and resize to the original length.
    """
    def __init__(self, lower_bnd=0.75, upper_bnd=0.75, fixed_crop_len=None, start_idx=None, resize=False, fixed_resize_len=None) -> None:
        self.lower_bnd = lower_bnd
        self.upper_bnd = upper_bnd
        self.fixed_crop_len = fixed_crop_len
        self.start_idx = start_idx
        self.resize = resize
        self.fixed_resize_len = fixed_resize_len

    def __call__(self, sample) -> Any:
        sample_dims = sample.dim()
        
        # define crop size
        if self.fixed_crop_len is not None:
            crop_len = self.fixed_crop_len
        else:
            # randomly sample the target length from a uniform distribution
            crop_len = int(sample.shape[-1]*np.random.uniform(low=self.lower_bnd, high=self.upper_bnd))
        
        # define cut-off point
        if self.start_idx is not None:
            start_idx = self.start_idx
        else:
            # randomly sample the starting point for the cropping (cut-off)
            try:
                start_idx = np.random.randint(low=0, high=sample.shape[-1]-crop_len)
            except ValueError:
                # if sample.shape[-1]-crop_len == 0, np.random.randint() throws an error
                start_idx = 0 


        # crop and resize the signal
        if self.resize == True:
            # define length after resize operation
            if self.fixed_resize_len is not None:
                resize_len = self.fixed_resize_len
            else:
                resize_len = sample.shape[-1]

            # crop and resize the signal
            cropped_sample = torch.zeros_like(sample[..., :resize_len])
            if sample_dims == 2:
                for ch in range(sample.shape[-2]):
                    resized_signal = np.interp(np.linspace(0, crop_len, num=resize_len), np.arange(crop_len), sample[ch, start_idx:start_idx+crop_len])
                    cropped_sample[ch, :] = torch.from_numpy(resized_signal)
            elif sample_dims == 3:
                for f_bin in range(sample.shape[-3]):
                    for ch in range(sample.shape[-2]):
                        resized_signal = np.interp(np.linspace(0, crop_len, num=resize_len), np.arange(crop_len), sample[f_bin, ch, start_idx:start_idx+crop_len])
                        cropped_sample[f_bin, ch, :] = torch.from_numpy(resized_signal)
            else:
                sys.exit('Error. Sample dimension does not match.')
        else:
            # only crop the signal
            cropped_sample = torch.zeros_like(sample)
            cropped_sample = sample[..., start_idx:start_idx+crop_len]

        return cropped_sample

class Interpolation(object):
    """
        Undersample the signal and interpolate to initial length.
    """
    def __init__(self, step=2) -> None:
        self.step = step

    def __call__(self, sample) -> Any:
        sample_sub = sample[..., ::self.step]
        sample_interpolated = np.ones_like(sample)
        
        sample_dims = sample.dim()
        if sample_dims == 2:
            for ch in range(sample.shape[-2]):
                sample_interpolated[ch] = np.interp(np.arange(0, sample.shape[-1]), np.arange(0, sample.shape[-1], step=self.step), sample_sub[ch])
        elif sample_dims == 3:
            for f_bin in range(sample.shape[-3]):
                for ch in range(sample.shape[-2]):
                    sample_interpolated[f_bin, ch] = np.interp(np.arange(0, sample.shape[-1]), np.arange(0, sample.shape[-1], step=self.step), sample_sub[f_bin, ch])
        else:
            sys.exit('Error. Sample dimension does not match.')

        return torch.from_numpy(sample_interpolated)

class Masking_(object):
    """
        Randomly zero-mask the sample.
    """
    def __init__(self, factor=0.25, fs=250, patch_size_sec=5) -> None:
        self.factor = factor                    # fraction to be masked out
        self.patch_size = patch_size_sec * fs   # patch_size[ticks] = patch_size[sec] * fs[Hz]

    def __call__(self, sample) -> Any:
        # determine the number of patches to be masked
        nb_patches = round(self.factor * sample.shape[-1] / self.patch_size)
 
        mask = torch.ones_like(sample)

        for ch in range(sample.shape[-2]):
            # starting points of the patches to be masked
            indices = torch.randint(low=0, high=sample.shape[-1]-self.patch_size, size=(nb_patches,))

            # mask the signal
            for idx in indices:
                mask[..., ch, idx:idx+self.patch_size] = 0

        return sample * mask

class Masking(object):
    """
        Randomly zero-mask the sample.
        Got this from https://stackoverflow.com/questions/70092136/how-do-i-create-a-random-mask-matrix-where-we-mask-a-contiguous-length
        Don't touch the code!
    """
    def __init__(self, factor=0.25, fs=250, patch_size_sec=5) -> None:
        self.factor = factor                    # fraction to be masked out
        self.patch_size = patch_size_sec * fs   # patch_size[ticks] = patch_size[sec] * fs[Hz]

    def __call__(self, sample) -> Any:
        # create the mask
        mask = torch.ones_like(sample)

        # determine the number of patches to be masked
        nb_patches = round(self.factor * sample.shape[-1] / self.patch_size)
        

        indices_weights = np.random.random((mask.shape[0], nb_patches + 1))

        number_of_ones = mask.shape[-1] - self.patch_size * nb_patches

        ones_sizes = np.round(indices_weights[:, :nb_patches].T
                            * (number_of_ones / np.sum(indices_weights, axis=-1))).T.astype(np.int32)

        ones_sizes[:, 1:] += self.patch_size

        zeros_start_indices = np.cumsum(ones_sizes, axis=-1)

        for sample_idx in range(len(mask)):
            for zeros_idx in zeros_start_indices[sample_idx]:
                mask[sample_idx, zeros_idx: zeros_idx + self.patch_size] = 0

        return sample * mask, 1-mask


class Masking(object):
    """
        Randomly zero-mask the sample.
        Got this from https://stackoverflow.com/questions/70092136/how-do-i-create-a-random-mask-matrix-where-we-mask-a-contiguous-length
        Don't touch the code!
    """
    def __init__(self, factor=0.75, fs=200, patch_size_sec=1) -> None:
        self.factor = factor                    # fraction to be masked out
        self.patch_size = patch_size_sec * fs   # patch_size[ticks] = patch_size[sec] * fs[Hz]

    def __call__(self, sample) -> Any:
        # create the mask
        mask = torch.ones_like(sample)

        # determine the number of patches to be masked
        nb_patches = round(self.factor * sample.shape[-1] / self.patch_size)
        

        indices_weights = np.random.random((mask.shape[0], nb_patches + 1))

        number_of_ones = mask.shape[-1] - self.patch_size * nb_patches

        ones_sizes = np.round(indices_weights[:, :nb_patches].T
                            * (number_of_ones / np.sum(indices_weights, axis=-1))).T.astype(np.int32)

        ones_sizes[:, 1:] += self.patch_size

        zeros_start_indices = np.cumsum(ones_sizes, axis=-1)

        #for sample_idx in range(len(mask)):
        for zeros_idx in zeros_start_indices[0]:
            mask[..., zeros_idx: zeros_idx + self.patch_size] = 0

        return sample * mask