"""
Copyright (c) 2025 Bahareh Tolooshams

loss functions for training

:author: Bahareh Tolooshams
"""

import torch
import pytorch_msssim


class FUSLoss(object):
    def __init__(
        self,
        ssim_alpha=0.1,
        mse_alpha=0.4,
        mae_alpha=0.5,
        data_range=1,
        num_channel=1,
        ssim_flag=True,
        ssim_win_size=7,
        ssim_normalize_data_into_range=False,
        ssim_normalize_together=False,
        ssim_divide_by=1,
    ) -> None:
        self.ssim_alpha = ssim_alpha
        self.mse_alpha = mse_alpha
        self.mae_alpha = mae_alpha
        self.data_range = data_range  # 1 or 255
        self.num_channel = num_channel
        self.ssim_flag = ssim_flag
        self.ssim_win_size = ssim_win_size
        self.ssim_normalize_data_into_range = ssim_normalize_data_into_range
        self.ssim_normalize_together = ssim_normalize_together
        self.ssim_divide_by = ssim_divide_by

        # data format for this criterion is (batch, channels, H, W)
        if self.ssim_flag:
            self.ssim_criterion = pytorch_msssim.SSIM(
                data_range=self.data_range,
                size_average=True,
                channel=self.num_channel,
                win_size=self.ssim_win_size,
            )

    def normalize_signal(self, x):
        batch, ch, h, w = x.shape
        x_flat = x.reshape(batch, -1)
        x_flat = torch.div(
            x_flat - torch.min(x_flat, dim=-1, keepdim=True)[0],
            torch.max(x_flat, dim=-1, keepdim=True)[0]
            - torch.min(x_flat, dim=-1, keepdim=True)[0],
        )
        x = x_flat.reshape(batch, ch, h, w)
        return x

    def normalize_signal_with_given_minmax(self, x, a, b):
        x = torch.div(x - a, b - a)
        return x

    def fus_loss(self, x, y, eps=1e-5):
        mae_loss = torch.mean(torch.abs(x - y))
        mse_loss = torch.mean((x - y) ** 2)

        if self.ssim_flag:
            if self.ssim_normalize_data_into_range:
                # the inputs to ssim should have the range of [0,1] or [0, 255] (see data range for 1 or 255)
                if self.ssim_normalize_together:
                    a = torch.minimum(torch.min(x), torch.min(y))
                    b = torch.maximum(torch.max(x), torch.max(y))
                    x_for_ssim = self.normalize_signal_with_given_minmax(x, a, b)
                    y_for_ssim = self.normalize_signal_with_given_minmax(y, a, b)
                else:
                    x_for_ssim = self.normalize_signal(x)
                    y_for_ssim = self.normalize_signal(y)
            else:
                x_for_ssim = x / self.ssim_divide_by
                y_for_ssim = y / self.ssim_divide_by

            ssim_loss = 1.0 - self.ssim_criterion(x_for_ssim, y_for_ssim)

            loss = (
                self.ssim_alpha * ssim_loss
                + self.mae_alpha * mae_loss
                + self.mse_alpha * mse_loss
            )
        else:
            loss = self.mae_alpha * mae_loss + self.mse_alpha * mse_loss

        return loss

    def __call__(self, x, y):
        return self.fus_loss(x, y)
