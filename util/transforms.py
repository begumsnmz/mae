import sys

import torch
import torch.nn.functional as F
import torch.fft as fft


class Normalization(object):
    """
        Normalize the data.
    """
    def __call__(self, sample) -> torch.Tensor:
        mean = sample.mean(axis=-1)
        std = sample.std(axis=-1)

        # transpose to match the vector dimensions
        return ((sample.permute(dims=(2, 0, 1)) - mean) / std).permute(dims=(1, 2, 0))


class MinMaxScaling(object):
    """
        Scale the data to a range from [lower, upper].    
    """
    def __init__(self, lower=-1, upper=1) -> None:
        self.lower = lower
        self.upper = upper

    def __call__(self, sample) -> torch.Tensor:
        min = sample.min(axis=-1)[0]
        max = sample.max(axis=-1)[0]

        return ((sample.permute(dims=(1, 0)) - min) / (max - min)).permute(dims=(1, 0)) * (self.upper - self.lower) + self.lower


class OneHotEncoding(object):
    """
        Convert categorical targets into one hot encoded targets.
    """
    def __init__(self, nb_classes) -> None:
        super().__init__()
        self.nb_classes = nb_classes

    def __call__(self, label) -> torch.Tensor:
        return F.one_hot(label, num_classes=self.nb_classes).float()


class ArrayToTensor(object):
    """
        Convert ndarrays into tensors.
    """
    def __call__(self, sample) -> torch.Tensor:
        return torch.from_numpy(sample).to(torch.float32)


class ScalarToTensor(object):
    """
        Convert int into tensor.
    """
    def __call__(self, label) -> torch.Tensor:
        return torch.tensor(label)


class Ideal_filtering(object):
    """ 
        Remove certain frequency bins from the data.
        Ideal window is used for filtering.
    """
    def __init__(self, fs:int=250, f_0:int=100, band_width:int=5, mode:str="low_pass") -> None:
        super().__init__()
        self.fs = fs
        self.f_0 = f_0
        self.band_width = band_width
        self.mode = mode

    def __call__(self, sample) -> torch.Tensor:
        factor = 2
        N = factor * sample.shape[-1] # ouput length of the Fourier transform
        X_f = 1/N * fft.fftshift(fft.fft(sample, n=N)) # Factor 1/N is not crucial, only for visualization
        
        center = int(0.5*N)
        offset = int(self.f_0*(N/self.fs))

        if self.mode == "low_pass":
            X_f[:,:center-offset] = 0
            X_f[:,center+offset:] = 0
        elif self.mode == "high_pass":
            X_f[:,center-offset:center+offset] = 0
        elif self.mode == "band_stop":
            band_half = int(0.5*self.band_width*(N/self.fs))
            X_f[:,center-offset-band_half:center-offset+band_half] = 0
            X_f[:,center+offset-band_half:center+offset+band_half] = 0
        elif self.mode == "band_pass":
            band_half = int(0.5*self.band_width*(N/self.fs))
            X_f[:,:center-offset-band_half] = 0
            X_f[:,center-offset+band_half:center+offset-band_half] = 0
            X_f[:,center+offset+band_half:] = 0
        else:
            sys.exit("Error: Mode does not exist.")

        x_t = N * fft.ifft(fft.ifftshift(X_f), n=N)
        
        x_t = x_t[:,:int(N/factor)]

        return torch.real(x_t)


class Gaussian_filtering(object):
    """ 
        Remove certain frequency bins from the data.
        Gaussian window is used for filtering. 
    """
    def __init__(self, fs:int=250, f_0:int=100, band_width:int=5, mode:str="low_pass") -> None:
        super().__init__()
        self.fs = fs
        self.f_0 = f_0
        self.band_width = band_width
        self.mode = mode

    def __call__(self, sample) -> torch.Tensor:
        factor = 2
        N = factor * sample.shape[-1] # ouput length of the Fourier transform
        
        X_f = 1/N * fft.fftshift(fft.fft(sample, n=N))
        f = torch.arange(-N/2, N/2) * 1/N * self.fs

        if self.mode == "low_pass":
            std = 1.5 * self.f_0
            Filter = torch.exp(-(f/(std)).pow(2))
        elif self.mode == "high_pass":
            std = 2 * self.f_0
            Filter = 1 - torch.exp(-(f/(std)).pow(2))
        elif self.mode == "band_stop":
            std = self.band_width
            Filter = 1 - ( torch.exp(-((f+self.f_0)/(std)).pow(2)) + torch.exp(-((f-self.f_0)/(std)).pow(2)) )
        elif self.mode == "band_pass":
            std = self.band_width
            Filter = torch.exp(-((f+self.f_0)/(std)).pow(2)) + torch.exp(-((f-self.f_0)/(std)).pow(2)) 
        else:
            sys.exit("Error: Mode does not exist.")

        x_t = N * fft.ifft(fft.ifftshift(Filter*X_f), n=N)
        
        x_t = x_t[:,:int(N/factor)]

        return torch.real(x_t)