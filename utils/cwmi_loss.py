import torch
import torchvision
from .loss import *

from PIL import Image
import matplotlib.pyplot as plt
import math

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

_POS_ALPHA = 5e-4





class CWMI_loss(torch.nn.Module):
    def __init__(self, complex, cw, spN = 4, spK=4, lamb=0.9, CW_method="MI", select = None):
        # CW_method: MI: mutual information; L1: L1 distance; L2: L2 distance; SSIM: Structure SIMilarity
        super(CWMI_loss, self).__init__()
        self.sp = ComplexSteerablePyramid(complex=complex, N=spN, K=spK)
        self.complex = complex
        self.lamb = lamb
        self.BCEW = BCE_withClassBalance()
        self.CW_method = CW_method
        if self.CW_method == "SSIM":
            self.ssim = SSIM()
        self.select = select
        self.cw = cw
        #self.block_complex = block_complex

    def forward(self, pred, mask, epoch=None):
        if epoch == 0:
            return self.BCEW(mask, pred, None, self.cw)
        sp_mask = self.sp(mask)
        sp_pred = self.sp(pred)
        mi_output = []
        for i in range(self.sp.N):
            if self.CW_method == "MI":
                if self.complex:
                    mi_output.append(torch.mean(self.complex_mi(sp_mask[i + 1], sp_pred[i + 1])).real)
                else:
                    mi_output.append(torch.mean(self.real_mi(sp_mask[i + 1], sp_pred[i + 1])))
            elif self.CW_method == "L1":
                mi_output.append(torch.mean(torch.sum(torch.abs(sp_mask[i + 1] - sp_pred[i + 1]), dim=2)))
            elif self.CW_method == "L2":
                mi_output.append(torch.mean(torch.sqrt(torch.sum(torch.pow(torch.abs(sp_mask[i + 1] - sp_pred[i + 1]), 2), dim=2))))
            elif self.CW_method == "SSIM":
                if self.complex:
                    mi_output.append(self.ssim(torch.abs(sp_mask[i + 1]).squeeze(1), torch.abs(sp_pred[i + 1]).squeeze(1)))
                else:
                    mi_output.append(self.ssim(sp_mask[i + 1].squeeze(1), sp_pred[i + 1].squeeze(1)))

        loss = self.BCEW(mask, pred, None, self.cw) * self.lamb
        if self.select == None:
            for i in range(self.sp.N):
                loss += mi_output[i]
        else:
            loss += mi_output[self.select]
        return loss

    def real_mi(self, mask, pred):
        B, C, A, H, W = mask.shape # A: angle, number of orientations of the steerable pyramid
        mask_flat = mask.view(B, C, A, H * W).type(torch.cuda.DoubleTensor)
        mask_mean = torch.mean(mask_flat, dim=3, keepdim=True)
        mask_centered = mask_flat - mask_mean

        pred_flat = pred.view(B, C, A, H * W).type(torch.cuda.DoubleTensor)
        pred_mean = torch.mean(pred_flat, dim=3, keepdim=True)
        pred_centered = pred_flat - pred_mean

        var_mask = torch.matmul(mask_centered, torch.permute(mask_centered, (0, 1, 3, 2)))
        var_pred = torch.matmul(pred_centered, torch.permute(pred_centered, (0, 1, 3, 2)))
        cov_mask_pred = torch.matmul(mask_centered, torch.permute(pred_centered, (0, 1, 3, 2)))

        diag_matrix = torch.eye(A)
        inv_cov_pred = torch.inverse(var_pred + diag_matrix.type_as(var_pred) * _POS_ALPHA)

        cond_cov_mask_pred = var_mask - torch.matmul(torch.matmul(cov_mask_pred, inv_cov_pred), torch.permute(cov_mask_pred, (0, 1, 3, 2)))

        chol = torch.linalg.cholesky(cond_cov_mask_pred)
        return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)

    def complex_mi(self, mask, pred):
        B, C, A, H, W = mask.shape # A: angle, number of orientations of the steerable pyramid
        mask_flat = mask.view(B, C, A, H * W)
        mask_flat = torch.cat((torch.cat((mask_flat.real, mask_flat.imag), dim=2), torch.cat((-mask_flat.imag, mask_flat.real), dim=2)), dim=3)
        mask_mean = torch.mean(mask_flat, dim=3, keepdim=True)
        mask_centered = mask_flat - mask_mean

        pred_flat = pred.view(B, C, A, H * W)
        pred_flat = torch.cat((torch.cat((pred_flat.real, pred_flat.imag), dim=2), torch.cat((-pred_flat.imag, pred_flat.real), dim=2)), dim=3)
        pred_mean = torch.mean(pred_flat, dim=3, keepdim=True)
        pred_centered = pred_flat - pred_mean

        var_mask = torch.matmul(mask_centered, torch.permute(mask_centered, (0, 1, 3, 2)))
        var_pred = torch.matmul(pred_centered, torch.permute(pred_centered, (0, 1, 3, 2)))
        cov_mask_pred = torch.matmul(mask_centered, torch.permute(pred_centered, (0, 1, 3, 2)))

        diag_matrix = torch.eye(2*A)
        inv_cov_pred = torch.inverse(var_pred + diag_matrix.type_as(var_pred) * _POS_ALPHA)

        cond_cov_mask_pred = var_mask - torch.matmul(torch.matmul(cov_mask_pred, inv_cov_pred), torch.permute(cov_mask_pred, (0, 1, 3, 2)))

        chol = torch.linalg.cholesky(cond_cov_mask_pred)
        return torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)


class ComplexSteerablePyramid(torch.nn.Module):
    def __init__(self, complex=False, N=4, K=12, device='cuda'):
        super(ComplexSteerablePyramid, self).__init__()
        self.N = N
        self.K = K
        self.complex = complex
        self.masks = {}
        self.device = device

    def get_grid(self, H, W):
        x = torch.linspace(-(H // 2 - 1) * np.pi / (H // 2), np.pi, H)
        x = x.reshape((H, 1)).expand((H, W))
        y = torch.linspace(-(W // 2 - 1) * np.pi / (W // 2), np.pi, W)
        y = y.reshape((1, W)).expand((H, W))
        radius = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.arctan(y / x)
        theta[x < 0] += torch.pi
        theta[(x >=0) & (y < 0)] += torch.pi * 2
        return radius, theta
    
    def down_sample(self, fourier_domain_image):
        B, C, H, W = fourier_domain_image.shape
        return fourier_domain_image[:, :, H // 4 : H // 4 * 3, W // 4 : W // 4 * 3]
    
    def up_sample(self, fourier_domain_image):
        B, C, H, W = fourier_domain_image.shape
        output = torch.zeros((B, C, 2 * H, 2 * W)).to(self.device).to(fourier_domain_image.dtype)
        output[:, :, H // 2: H // 2 * 3, W // 2: W // 2 * 3] = fourier_domain_image
        return output

    def get_mask(self, image_size: tuple[int]):
        if image_size in self.masks:
            return self.masks[image_size]
        high_pass_filters = []
        low_pass_filters = []
        band_filters = []

        H, W = image_size
        for i in range(self.N + 1):
            if i == 0:
                radius, theta = self.get_grid(H, W)
                high_i = torch.zeros_like(radius)
                high_i[(radius > torch.pi / 2) & (radius < torch.pi)] = \
                    torch.cos(torch.pi / 2 * torch.log2(radius[(radius > torch.pi / 2) & (radius < torch.pi)] / torch.pi))
                high_i[radius >= torch.pi] = 1
                high_pass_filters.append(high_i.to(self.device))
                low_i = torch.zeros_like(radius)
                low_i[(radius > torch.pi / 2) & (radius < torch.pi)] = \
                    torch.cos(torch.pi / 2 * torch.log2(2 * radius[(radius > torch.pi / 2) & (radius < torch.pi)] / torch.pi))
                low_i[radius <= torch.pi / 2] = 1
                low_pass_filters.append(low_i.to(self.device))
                band_filters.append(None)
            else:
                radius, theta = self.get_grid(H, W)
                high_i = torch.zeros_like(radius)
                high_i[(radius > torch.pi / 4) & (radius < torch.pi / 2)] = \
                    torch.cos(torch.pi / 2 * torch.log2(2 * radius[(radius > torch.pi / 4) & (radius < torch.pi / 2)] / torch.pi))
                high_i[radius >= torch.pi / 2] = 1
                high_pass_filters.append(high_i.to(self.device))
                low_i = torch.zeros_like(radius)
                low_i[(radius > torch.pi / 4) & (radius < torch.pi / 2)] = \
                    torch.cos(torch.pi / 2 * torch.log2(4 * radius[(radius > torch.pi / 4) & (radius < torch.pi / 2)] / torch.pi))
                low_i[radius <= torch.pi / 4] = 1
                low_pass_filters.append(low_i.to(self.device))
                band_i = torch.zeros((self.K, H, W))
                alpha_k = 2 ** (self.K - 1) * math.factorial(self.K - 1) / math.sqrt(self.K * math.factorial(2 * (self.K - 1)))
                for k in range(self.K):
                    if self.complex:
                        band_i[k] = 2 * torch.abs(alpha_k * torch.pow(torch.nn.ReLU()(torch.cos(theta - torch.pi * k / self.K)), self.K - 1))
                    else:
                        band_i[k] = torch.abs(alpha_k * torch.pow(torch.cos(theta - torch.pi * k / self.K), self.K - 1))
                    band_i[k, H // 2 - 1, W // 2 - 1] = 0
                band_filters.append(band_i.to(self.device))
                H = H // 2
                W = W // 2
        self.masks[image_size] = {"high": high_pass_filters, "low": low_pass_filters, "band": band_filters}
        return self.masks[image_size]
    
    def forward(self, batch_images):
        output = []
        B, C, H, W = batch_images.shape
        masks = self.get_mask((H, W))
        fourier_domain = torch.fft.fftshift(torch.fft.fft2(batch_images), dim=(2, 3))
        high_freq_residue = fourier_domain * masks['high'][0]
        output.append(torch.fft.ifft2(torch.fft.ifftshift(high_freq_residue, dim=(2, 3))))
        fourier_domain = fourier_domain * masks['low'][0]
        for i in range(1, self.N + 1):
            band_signal = fourier_domain.unsqueeze(2) * masks['high'][i] * masks['band'][i]
            output.append(torch.fft.ifft2(torch.fft.ifftshift(band_signal, dim=(3, 4))))
            fourier_domain = fourier_domain * masks['low'][i]
            fourier_domain = self.down_sample(fourier_domain)
        output.append(torch.fft.ifft2(torch.fft.ifftshift(fourier_domain, dim=(2,3))))
        if self.complex:
            return output
        else:
            return [o.real for o in output]
        
    def reconstruct(self, image):
        B, C, H, W = image[0].shape
        masks = self.get_mask((H, W))
        if image[0].is_complex():
            fourier_domain = [torch.fft.fftshift(torch.fft.fft2(i.real), dim=(-2, -1)) for i in image]
        else:
            fourier_domain = [torch.fft.fftshift(torch.fft.fft2(i), dim=(-2, -1)) for i in image]
        output = fourier_domain[-1]
        for i in range(self.N, 0, -1):
            output = self.up_sample(output) * masks['low'][i]
            output += torch.sum(fourier_domain[i] * masks['band'][i], dim=2) * masks['high'][i]
        output = output * masks['low'][0] + fourier_domain[0] * masks['high'][0]
        return torch.fft.ifft2(torch.fft.ifftshift(output, dim=(2,3))).real
    

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

