"""
Copyright (c) 2025 Tolooshams

utils

:author: Bahareh Tolooshams
"""

import torch
import torchvision
import numpy as np
from tqdm import tqdm


def get_standardization_constant(params):
    if params.standardization_constant_path:
        standardization_constant = torch.load(params.standardization_constant_path)
        # only to scale all by one scalar value
        if params.standardize_iq_by_scalar:
            standardization_constant["iq_mean"] = torch.mean(
                standardization_constant["iq_mean"]
            )
            standardization_constant["iq_std"] = torch.mean(
                standardization_constant["iq_std"]
            )
        # only to scale all by one scalar value
        if params.standardize_dop_by_scalar:
            standardization_constant["dop_mean"] = torch.mean(
                standardization_constant["dop_mean"]
            )
            standardization_constant["dop_std"] = torch.mean(
                standardization_constant["dop_std"]
            )
    else:
        standardization_constant = None

    return standardization_constant


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, p=0.5, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return x + torch.randn(x.size(), device=x.device) * self.std + self.mean

        return x


class AddRandomMasking(torch.nn.Module):
    def __init__(self, p=0.5, mask_prob_range=[0.1, 0.3]):
        super().__init__()

        self.p = p
        self.mask_prob_range = mask_prob_range

    def generate_random_masking(self, x):
        # input is (1, h, w)

        x_t, x_h, x_w = x.shape
        prob = np.random.uniform(self.mask_prob_range[0], self.mask_prob_range[1])

        mask_vec = torch.ones([1, x_h * x_w])
        samples = np.random.choice(x_h * x_w, int(x_w * x_h * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, x_h, x_w)
        mask_b = mask_b.repeat(x_t, 1, 1)
        mask = torch.ones_like(x, device=x.device)
        mask[:, ...] = mask_b
        return mask

    def forward(self, x):
        if torch.rand(1) < self.p:
            return x
        else:
            # (t,h,w)
            mask_x = self.generate_random_masking(x)

            return mask_x * x


class FUSRandomMasking(torch.nn.Module):
    def __init__(self, p=0.5, mask_prob_range=[0.1, 0.3]):
        super().__init__()

        self.p = p
        self.mask_prob_range = mask_prob_range

    def generate_random_masking(self, x):
        # input is (t, h, w)

        x_t, x_h, x_w = x.shape
        prob = np.random.uniform(self.mask_prob_range[0], self.mask_prob_range[1])

        mask_vec = torch.ones([1, x_h * x_w])
        samples = np.random.choice(x_h * x_w, int(x_w * x_h * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, x_h, x_w)
        mask_b = mask_b.repeat(x_t, 1, 1)
        mask = torch.ones_like(x, device=x.device)
        mask[:, ...] = mask_b
        return mask

    def forward(self, iq, dop):
        if torch.rand(1) < self.p:
            return iq, dop
        else:
            # (t,h,w)
            mask_iq = self.generate_random_masking(iq)

            # only mask the input iq, not dop

            iq = mask_iq * iq

            return iq, dop


class FUSRandomRotation(torchvision.transforms.RandomRotation):
    def __init__(self, degrees, p=0.5):
        super().__init__(degrees=degrees)

        self.p = p

    def forward(self, iq, dop, dop_from_svd=None):
        if torch.rand(1) < self.p:
            degrees = self.degrees
        else:
            degrees = [f + 90 for f in self.degrees]

        dop = dop.unsqueeze(dim=0)
        if isinstance(dop_from_svd, torch.Tensor):
            dop_from_svd = dop_from_svd.unsqueeze(dim=0)

        fill = self.fill
        channels_iq, _, _ = torchvision.transforms.functional.get_dimensions(iq)
        channels_dop, _, _ = torchvision.transforms.functional.get_dimensions(dop)
        if isinstance(dop_from_svd, torch.Tensor):
            (
                channels_dop_from_svd,
                _,
                _,
            ) = torchvision.transforms.functional.get_dimensions(dop_from_svd)

        if isinstance(iq, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill_iq = [float(fill)] * channels_iq
            else:
                fill_iq = [float(f) for f in fill]

        if isinstance(dop, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill_dop = [float(fill)] * channels_dop
            else:
                fill_dop = [float(f) for f in fill]

        if isinstance(dop_from_svd, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill_dop_from_svd = [float(fill)] * channels_dop_from_svd
            else:
                fill_dop_from_svd = [float(f) for f in fill]

        angle = self.get_params(degrees)

        if torch.is_complex(iq):
            # if complex, to real/imag separately
            iq_real = torchvision.transforms.functional.rotate(
                torch.real(iq),
                angle,
                self.interpolation,
                self.expand,
                self.center,
                fill_iq,
            )
            iq_imag = torchvision.transforms.functional.rotate(
                torch.imag(iq),
                angle,
                self.interpolation,
                self.expand,
                self.center,
                fill_iq,
            )
            iq = torch.complex(iq_real, iq_imag)
        else:
            iq = torchvision.transforms.functional.rotate(
                iq, angle, self.interpolation, self.expand, self.center, fill_iq
            )
        dop = torchvision.transforms.functional.rotate(
            dop, angle, self.interpolation, self.expand, self.center, fill_dop
        )

        dop = dop.squeeze(dim=0)

        if isinstance(dop_from_svd, torch.Tensor):
            dop_from_svd = torchvision.transforms.functional.rotate(
                dop_from_svd,
                angle,
                self.interpolation,
                self.expand,
                self.center,
                fill_dop_from_svd,
            )

            dop_from_svd = dop_from_svd.squeeze(dim=0)

        if isinstance(dop_from_svd, torch.Tensor):
            return iq, dop, dop_from_svd
        else:
            return iq, dop


class FUSRandomResizedCrop(torchvision.transforms.RandomResizedCrop):
    def __init__(
        self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), antialias=True
    ):
        super().__init__(
            size=size, scale=tuple(scale), ratio=tuple(ratio), antialias=antialias
        )

    def forward(self, iq, dop, dop_from_svd=None):
        i, j, h, w = self.get_params(iq, self.scale, self.ratio)

        dop = dop.unsqueeze(dim=0)
        if isinstance(dop_from_svd, torch.Tensor):
            dop_from_svd = dop_from_svd.unsqueeze(dim=0)

        iq_transform = torchvision.transforms.functional.resized_crop(
            iq, i, j, h, w, self.size, self.interpolation, antialias=self.antialias
        )
        dop_transform = torchvision.transforms.functional.resized_crop(
            dop, i, j, h, w, self.size, self.interpolation, antialias=self.antialias
        )

        dop_transform = dop_transform.squeeze(dim=0)

        if isinstance(dop_from_svd, torch.Tensor):
            dop_transform_from_svd = torchvision.transforms.functional.resized_crop(
                dop_from_svd,
                i,
                j,
                h,
                w,
                self.size,
                self.interpolation,
                antialias=self.antialias,
            )

            dop_transform_from_svd = dop_transform_from_svd.squeeze(dim=0)

        if isinstance(dop_from_svd, torch.Tensor):
            return iq_transform, dop_transform, dop_transform_from_svd
        else:
            return iq_transform, dop_transform


class FUSRandomCrop(torchvision.transforms.RandomCrop):
    def __init__(
        self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"
    ):
        super().__init__(
            size=size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode,
        )

    def forward(self, iq, dop, dop_from_svd=None):
        if self.padding is not None:
            iq = torchvision.transforms.functional.pad(
                iq, self.padding, self.fill, self.padding_mode
            )
            dop = torchvision.transforms.functional.pad(
                dop, self.padding, self.fill, self.padding_mode
            )

            if isinstance(dop_from_svd, torch.Tensor):
                dop_from_svd = torchvision.transforms.functional.pad(
                    dop_from_svd, self.padding, self.fill, self.padding_mode
                )

        _, height, width = torchvision.transforms.functional.get_dimensions(iq)
        # pad the width if needed
        if self.pad_if_needed:
            pad_leftright = np.maximum(self.size[1] - width, 0)
            pad_topbottom = np.maximum(self.size[0] - height, 0)

            pad_left = int(np.floor(np.random.rand(1) * pad_leftright))
            pad_right = pad_leftright - pad_left

            pad_top = int(np.floor(np.random.rand(1) * pad_topbottom))
            pad_bottom = pad_topbottom - pad_top

            padding = [pad_left, pad_top, pad_right, pad_bottom]

            iq = torchvision.transforms.functional.pad(
                iq, padding, self.fill, self.padding_mode
            )
            dop = torchvision.transforms.functional.pad(
                dop, padding, self.fill, self.padding_mode
            )
            if isinstance(dop_from_svd, torch.Tensor):
                dop_from_svd = torchvision.transforms.functional.pad(
                    dop_from_svd, padding, self.fill, self.padding_mode
                )

        i, j, h, w = self.get_params(iq, self.size)

        if isinstance(dop_from_svd, torch.Tensor):
            return (
                torchvision.transforms.functional.crop(iq, i, j, h, w),
                torchvision.transforms.functional.crop(dop, i, j, h, w),
                torchvision.transforms.functional.crop(dop_from_svd, i, j, h, w),
            )
        else:
            return torchvision.transforms.functional.crop(
                iq, i, j, h, w
            ), torchvision.transforms.functional.crop(dop, i, j, h, w)


class FUSAddGaussianNoise(torch.nn.Module):
    def __init__(self, p=0.5, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, iq, dop, dop_from_svd=None):
        if isinstance(dop_from_svd, torch.Tensor):
            if torch.rand(1) < self.p:
                return (
                    iq
                    + torch.randn(iq.size(), device=iq.device) * self.std
                    + self.mean,
                    dop,
                    dop_from_svd,
                )
            return iq, dop, dop_from_svd
        else:
            if torch.rand(1) < self.p:
                return (
                    iq
                    + torch.randn(iq.size(), device=iq.device) * self.std
                    + self.mean,
                    dop,
                )
            return iq, dop


class FUSRandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, iq, dop, dop_from_svd=None):
        if isinstance(dop_from_svd, torch.Tensor):
            if torch.rand(1) < self.p:
                return (
                    torchvision.transforms.functional.hflip(iq),
                    torchvision.transforms.functional.hflip(dop),
                    torchvision.transforms.functional.hflip(dop_from_svd),
                )
            return iq, dop, dop_from_svd
        else:
            if torch.rand(1) < self.p:
                return torchvision.transforms.functional.hflip(
                    iq
                ), torchvision.transforms.functional.hflip(dop)
            return iq, dop


class FUSRandomVerticalFlip(torchvision.transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, iq, dop, dop_from_svd=None):
        if isinstance(dop_from_svd, torch.Tensor):
            if torch.rand(1) < self.p:
                return (
                    torchvision.transforms.functional.vflip(iq),
                    torchvision.transforms.functional.vflip(dop),
                    torchvision.transforms.functional.vflip(dop_from_svd),
                )
            return iq, dop, dop_from_svd
        else:
            if torch.rand(1) < self.p:
                return torchvision.transforms.functional.vflip(
                    iq
                ), torchvision.transforms.functional.vflip(dop)
            return iq, dop


class FUSCenterCrop(torchvision.transforms.CenterCrop):
    def __init__(self, size):
        super().__init__(size=size)

    def forward(self, iq, dop, dop_from_svd=None):
        if isinstance(dop_from_svd, torch.Tensor):
            return (
                torchvision.transforms.functional.center_crop(iq, self.size),
                torchvision.transforms.functional.center_crop(dop, self.size),
                torchvision.transforms.functional.center_crop(dop_from_svd, self.size),
            )
        else:
            return torchvision.transforms.functional.center_crop(
                iq, self.size
            ), torchvision.transforms.functional.center_crop(dop, self.size)


def standardize(
    iq_signal,
    dop_signal,
    standardization_constant,
    iq_signal_mode,
    dop_from_svd=None,
    eps=1e-5,
):
    if iq_signal_mode == "complex":
        iq_signal_abs = torch.abs(iq_signal)
        iq_signal_angle = torch.angle(iq_signal)
        iq_signal_abs = (iq_signal_abs - standardization_constant["iq_mean"]) / (
            standardization_constant["iq_std"] + eps
        )
        iq_signal = torch.polar(iq_signal_abs, iq_signal_angle)
    elif iq_signal_mode == "real" or iq_signal_mode == "abs":
        iq_signal = (iq_signal - standardization_constant["iq_mean"]) / (
            standardization_constant["iq_std"] + eps
        )
    elif iq_signal_mode == "stack":
        nt = int(iq_signal.shape[-1] / 2)

        if iq_signal.dim() == 3:
            iq_signal_real = iq_signal[:, :, :nt]
            iq_signal_imag = iq_signal[:, :, nt:]
        else:
            iq_signal_real = iq_signal[:, :, :, :nt]
            iq_signal_imag = iq_signal[:, :, :, nt:]

        iq_signal_real = (iq_signal_real - standardization_constant["iq_mean_real"]) / (
            standardization_constant["iq_std_real"] + eps
        )
        iq_signal_imag = (iq_signal_imag - standardization_constant["iq_mean_imag"]) / (
            standardization_constant["iq_std_imag"] + eps
        )

        iq_signal = torch.cat([iq_signal_real, iq_signal_imag], dim=-1)
    else:
        raise NotImplementedError

    dop_signal = (dop_signal - standardization_constant["dop_mean"]) / (
        standardization_constant["dop_std"] + eps
    )
    if isinstance(dop_from_svd, torch.Tensor):
        dop_from_svd = (dop_from_svd - standardization_constant["dop_mean"]) / (
            standardization_constant["dop_std"] + eps
        )
        return iq_signal, dop_signal, dop_from_svd
    else:
        return iq_signal, dop_signal


def unstandardize(
    iq_signal, dop_signal, standardization_constant, iq_signal_mode, dop_from_svd=None
):
    if iq_signal_mode == "complex":
        iq_signal_abs = torch.abs(iq_signal)
        iq_signal_angle = torch.angle(iq_signal)
        iq_signal_abs = (
            iq_signal_abs * standardization_constant["iq_std"]
            + standardization_constant["iq_mean"]
        )
        iq_signal = torch.polar(iq_signal_abs, iq_signal_angle)
    elif iq_signal_mode == "real" or iq_signal_mode == "abs":
        iq_signal = (
            iq_signal * standardization_constant["iq_std"]
            + standardization_constant["iq_mean"]
        )
    elif iq_signal_mode == "stack":

        nt = int(iq_signal.shape[-1] / 2)

        if iq_signal.dim() == 3:
            iq_signal_real = iq_signal[:, :, :nt]
            iq_signal_imag = iq_signal[:, :, nt:]
        else:
            iq_signal_real = iq_signal[:, :, :, :nt]
            iq_signal_imag = iq_signal[:, :, :, nt:]

        iq_signal_real = (
            iq_signal_real * standardization_constant["iq_std_real"]
            + standardization_constant["iq_mean_real"]
        )
        iq_signal_imag = (
            iq_signal_imag * standardization_constant["iq_std_imag"]
            + standardization_constant["iq_mean_imag"]
        )

        iq_signal = torch.cat([iq_signal_real, iq_signal_imag], dim=-1)
    else:
        raise NotImplementedError

    dop_signal = (
        dop_signal * standardization_constant["dop_std"]
        + standardization_constant["dop_mean"]
    )
    if isinstance(dop_from_svd, torch.Tensor):
        dop_from_svd = (
            dop_from_svd * standardization_constant["dop_std"]
            + standardization_constant["dop_mean"]
        )
        return iq_signal, dop_signal, dop_from_svd
    else:
        return iq_signal, dop_signal


def unstandardize_general(x, mean, std):
    device = x.device
    return x * std.to(device) + mean.to(device)


def standardize_general(x, mean, std, eps=1e-5):
    device = x.device
    return (x - mean.to(device)) / (std.to(device) + eps)


def compute_scalar_standardization_constant(
    dataset, iq_signal_mode, batch_size=128, num_workers=4
):
    # We take mean and std across exmaples and also pixels/time

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )

    # take dimensions from iq_signal
    h, w, t = dataset[0][0].size()
    num_data = len(dataset)

    dop_sum = torch.zeros(1)

    if iq_signal_mode == "stack":
        iq_sum_real = torch.zeros(1)
        iq_sum_imag = torch.zeros(1)
    else:
        iq_sum = torch.zeros(1)

    for idx, (iq_signal, dop_signal, _) in tqdm(enumerate(dataloader), disable=False):

        if iq_signal_mode == "stack":
            nt = int(iq_signal.shape[-1] / 2)

            if iq_signal.dim() == 3:
                iq_signal_real = iq_signal[:, :, :nt]
                iq_signal_imag = iq_signal[:, :, nt:]
            else:
                iq_signal_real = iq_signal[:, :, :, :nt]
                iq_signal_imag = iq_signal[:, :, :, nt:]

            # (batch_size * time frames * H * W)
            iq_signal_real = iq_signal_real.permute(0, 3, 1, 2)
            iq_signal_real = iq_signal_real.reshape(-1, 1)

            iq_signal_imag = iq_signal_imag.permute(0, 3, 1, 2)
            iq_signal_imag = iq_signal_imag.reshape(-1, 1)

        else:
            # (batch_size * time frames * H * W)
            iq_signal = iq_signal.permute(0, 3, 1, 2)
            iq_signal = iq_signal.reshape(-1, 1)

        # (batch_size * H * W)
        dop_signal = dop_signal.reshape(-1, 1)

        # if complex apply standarization on abs of the signal
        if iq_signal_mode == "complex":
            iq_signal_abs = torch.abs(iq_signal)
            iq_sum += iq_signal_abs.sum(0) / (t * h * w)
        elif iq_signal_mode == "real" or iq_signal_mode == "abs":
            iq_sum += iq_signal.sum(0) / (t * h * w)
        elif iq_signal_mode == "stack":
            iq_sum_real += iq_signal_real.sum(0) / (t * h * w)
            iq_sum_imag += iq_signal_imag.sum(0) / (t * h * w)

        else:
            raise NotImplementedError

        dop_sum += dop_signal.sum(0) / (h * w)

    if iq_signal_mode == "stack":
        iq_mean_real = iq_sum_real / num_data
        iq_mean_imag = iq_sum_imag / num_data
    else:
        iq_mean = iq_sum / num_data

    dop_mean = dop_sum / num_data

    if iq_signal_mode == "stack":
        iq_sum_sqrd_real = 0.0
        iq_sum_sqrd_imag = 0.0
    else:
        iq_sum_sqrd = 0.0
    dop_sum_sqrd = 0.0
    for idx, (iq_signal, dop_signal, _) in tqdm(enumerate(dataloader), disable=False):

        if iq_signal_mode == "stack":

            nt = int(iq_signal.shape[-1] / 2)

            if iq_signal.dim() == 3:
                iq_signal_real = iq_signal[:, :, :nt]
                iq_signal_imag = iq_signal[:, :, nt:]
            else:
                iq_signal_real = iq_signal[:, :, :, :nt]
                iq_signal_imag = iq_signal[:, :, :, nt:]

            # (batch_size * time frames * H * W)
            iq_signal_real = iq_signal_real.permute(0, 3, 1, 2)
            iq_signal_real = iq_signal_real.reshape(-1, 1)

            iq_signal_imag = iq_signal_imag.permute(0, 3, 1, 2)
            iq_signal_imag = iq_signal_imag.reshape(-1, 1)

        else:
            # (batch_size * time frames * H * W)
            iq_signal = iq_signal.permute(0, 3, 1, 2)
            iq_signal = iq_signal.reshape(-1, 1)

        # (batch_size * H * W)
        dop_signal = dop_signal.reshape(-1, 1)

        # if complex apply standarization on abs of the signal
        if iq_signal_mode == "complex":
            iq_signal_abs = torch.abs(iq_signal)
            iq_sum_sqrd += ((iq_signal_abs - iq_mean).pow(2)).sum(0) / (t * h * w)
        elif iq_signal_mode == "real" or iq_signal_mode == "abs":
            iq_sum_sqrd += ((iq_signal - iq_mean).pow(2)).sum(0) / (t * h * w)
        elif iq_signal_mode == "stack":
            iq_sum_sqrd_real += ((iq_signal_real - iq_mean_real).pow(2)).sum(0) / (
                t * h * w
            )
            iq_sum_sqrd_imag += ((iq_signal_imag - iq_mean_imag).pow(2)).sum(0) / (
                t * h * w
            )

        else:
            raise NotImplementedError

        dop_sum_sqrd += ((dop_signal - dop_mean).pow(2)).sum(0) / (h * w)

    if iq_signal_mode == "stack":
        iq_std_real = torch.sqrt(iq_sum_sqrd_real / num_data)
        iq_std_imag = torch.sqrt(iq_sum_sqrd_imag / num_data)
    else:
        iq_std = torch.sqrt(iq_sum_sqrd / num_data)

    dop_std = torch.sqrt(dop_sum_sqrd / num_data)

    standardization_constant = dict()
    if iq_signal_mode == "stack":
        standardization_constant["iq_mean"] = torch.zeros(1)  # this is placehodler
        standardization_constant["iq_std"] = torch.zeros(1)  # this is placeholder

        standardization_constant["iq_mean_real"] = iq_mean_real
        standardization_constant["iq_std_real"] = iq_std_real

        standardization_constant["iq_mean_imag"] = iq_mean_imag
        standardization_constant["iq_std_imag"] = iq_std_imag
    else:
        standardization_constant["iq_mean"] = iq_mean
        standardization_constant["iq_std"] = iq_std

    standardization_constant["dop_mean"] = dop_mean
    standardization_constant["dop_std"] = dop_std

    return standardization_constant


def compute_pixelwise_standardization_constant(
    dataset, iq_signal_mode, batch_size=128, num_workers=4
):
    # We take mean and std across exmaples.

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )

    # take dimensions from iq_signal
    h, w, t = dataset[0][0].size()
    num_data = len(dataset)

    iq_sum = torch.zeros(h, w)
    dop_sum = torch.zeros(h, w)

    for idx, (iq_signal, dop_signal, _) in tqdm(enumerate(dataloader), disable=False):
        # (batch_size * time frames, H, W)
        iq_signal = iq_signal.permute(0, 3, 1, 2)
        iq_signal = iq_signal.reshape(-1, iq_signal.shape[-2], iq_signal.shape[-1])

        # if complex apply standarization on abs of the signal
        if iq_signal_mode == "complex":
            iq_signal_abs = torch.abs(iq_signal)
            iq_sum += iq_signal_abs.sum(0) / t
        elif iq_signal_mode == "real" or iq_signal_mode == "abs":
            iq_sum += iq_signal.sum(0) / t
        else:
            raise NotImplementedError

        dop_sum += dop_signal.sum(0)

    iq_mean = iq_sum / num_data
    dop_mean = dop_sum / num_data

    iq_sum_sqrd = 0.0
    dop_sum_sqrd = 0.0
    for idx, (iq_signal, dop_signal, _) in tqdm(enumerate(dataloader), disable=False):
        # (batch_size * time frames, H, W)
        iq_signal = iq_signal.permute(0, 3, 1, 2)
        iq_signal = iq_signal.reshape(-1, iq_signal.shape[-2], iq_signal.shape[-1])

        # if complex apply standarization on abs of the signal
        if iq_signal_mode == "complex":
            iq_signal_abs = torch.abs(iq_signal)
            iq_sum_sqrd += ((iq_signal_abs - iq_mean).pow(2)).sum(0) / t
        elif iq_signal_mode == "real" or iq_signal_mode == "abs":
            iq_sum_sqrd += ((iq_signal - iq_mean).pow(2)).sum(0) / t
        else:
            raise NotImplementedError

        dop_sum_sqrd += ((dop_signal - dop_mean).pow(2)).sum(0)

    iq_std = torch.sqrt(iq_sum_sqrd / num_data)
    dop_std = torch.sqrt(dop_sum_sqrd / num_data)

    standardization_constant = dict()
    standardization_constant["iq_mean"] = iq_mean
    standardization_constant["iq_std"] = iq_std
    standardization_constant["dop_mean"] = dop_mean
    standardization_constant["dop_std"] = dop_std

    return standardization_constant


def compute_pixelwiseplustime_standardization_constant(
    dataset, iq_signal_mode, batch_size=128, num_workers=4
):
    # We take mean and std across exmaples.

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )

    # take dimensions from iq_signal
    h, w, t = dataset[0][0].size()
    num_data = len(dataset)

    iq_sum = torch.zeros(h, w, t)
    dop_sum = torch.zeros(h, w)

    for idx, (iq_signal, dop_signal, _) in tqdm(enumerate(dataloader), disable=False):
        # if complex apply standarization on abs of the signal
        if iq_signal_mode == "complex":
            iq_signal_abs = torch.abs(iq_signal)
            iq_sum += iq_signal_abs.sum(0)
        elif iq_signal_mode == "real" or iq_signal_mode == "abs":
            iq_sum += iq_signal.sum(0)
        else:
            raise NotImplementedError

        dop_sum += dop_signal.sum(0)

    iq_mean = iq_sum / num_data
    dop_mean = dop_sum / num_data

    iq_sum_sqrd = 0.0
    dop_sum_sqrd = 0.0
    for idx, (iq_signal, dop_signal, _) in tqdm(enumerate(dataloader), disable=False):
        # if complex apply standarization on abs of the signal
        if iq_signal_mode == "complex":
            iq_signal_abs = torch.abs(iq_signal)
            iq_sum_sqrd += ((iq_signal_abs - iq_mean).pow(2)).sum(0)
        elif iq_signal_mode == "real" or iq_signal_mode == "abs":
            iq_sum_sqrd += ((iq_signal - iq_mean).pow(2)).sum(0)
        else:
            raise NotImplementedError

        dop_sum_sqrd += ((dop_signal - dop_mean).pow(2)).sum(0)

    iq_std = torch.sqrt(iq_sum_sqrd / num_data)
    dop_std = torch.sqrt(dop_sum_sqrd / num_data)

    standardization_constant = dict()
    standardization_constant["iq_mean"] = iq_mean
    standardization_constant["iq_std"] = iq_std
    standardization_constant["dop_mean"] = dop_mean
    standardization_constant["dop_std"] = dop_std

    return standardization_constant
