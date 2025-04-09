"""
Copyright (c) 2025 Bahareh Tolooshams

metrics for evaluation

:author: Bahareh Tolooshams
"""

import torch
import numpy as np


def mae(y_true, y_pred):
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    loss = torch.mean(torch.abs(y_pred_flat - y_true_flat), dim=-1)
    return torch.mean(loss)


def nmae(y_true, y_pred, eps=1e-6):
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    loss = torch.mean(torch.abs(y_pred_flat - y_true_flat), dim=-1) / (
        torch.mean(torch.abs(y_true_flat), dim=-1) + eps
    )
    return torch.mean(loss)


def mse(y_true, y_pred):
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    loss = torch.mean(torch.pow(y_pred_flat - y_true_flat, 2), dim=-1)
    return torch.mean(loss)


def nmse(y_true, y_pred, eps=1e-6):
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    loss = torch.mean(torch.pow(y_pred_flat - y_true_flat, 2), dim=-1) / (
        torch.mean(torch.pow(y_true_flat, 2), dim=-1) + eps
    )
    return torch.mean(loss)


def cosine_sim(y_true, y_pred):
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    y_true_flat_normalized = y_true_flat / torch.norm(y_true_flat, dim=-1)
    y_pred_flat_normalized = y_pred_flat / torch.norm(y_pred_flat, dim=-1)

    sim = torch.sum(y_true_flat_normalized * y_pred_flat_normalized, dim=-1)

    return torch.mean(sim)


def psnr(y_true, y_pred, eps=1e-6):
    def standard_signal(x, dr):
        batch, ch, h, w = x.shape
        x_flat = x.reshape(batch, -1)
        x_flat = torch.div(
            x_flat - torch.min(x_flat, dim=-1, keepdim=True)[0],
            torch.max(x_flat, dim=-1, keepdim=True)[0]
            - torch.min(x_flat, dim=-1, keepdim=True)[0],
        )
        x = x_flat.reshape(batch, ch, h, w)
        x = torch.clip(x, np.power(10, -dr / 10), 1)
        x = torch.log(x) / torch.log(torch.tensor(10.0)) * torch.tensor(10.0)
        return (x + dr) / dr

    dr = 40  # dynamic range
    y_true = standard_signal(y_true, dr)
    y_pred = standard_signal(y_pred, dr)

    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    msef = torch.mean(torch.pow(y_pred_flat - y_true_flat, 2), dim=-1)
    maxf = torch.amax(y_true_flat, dim=-1)
    psnr = 20 * torch.log10(maxf) - 10 * torch.log10(msef)
    return torch.mean(psnr)


def psnr_stand_with_same_minmax(y_true, y_pred, eps=1e-6):
    def standard_signal_with_given_minmax(x, dr, a, b):
        x = torch.div(x - a, b - a)
        x = torch.clip(x, np.power(10, -dr / 10), 1)
        x = torch.log(x) / torch.log(torch.tensor(10.0)) * torch.tensor(10.0)
        return (x + dr) / dr

    dr = 40  # dynamic range
    a = torch.minimum(torch.min(y_true), torch.min(y_pred))
    b = torch.maximum(torch.max(y_true), torch.max(y_pred))
    y_true = standard_signal_with_given_minmax(y_true, dr, a, b)
    y_pred = standard_signal_with_given_minmax(y_pred, dr, a, b)

    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    msef = torch.mean(torch.pow(y_pred_flat - y_true_flat, 2), dim=-1)
    maxf = torch.amax(y_true_flat, dim=-1)
    psnr = 20 * torch.log10(maxf) - 10 * torch.log10(msef)
    return torch.mean(psnr)


def ssim(y_true, y_pred, ssim_criterion, eps=1e-6):
    def normalize_signal(x):
        batch, ch, h, w = x.shape
        x_flat = x.reshape(batch, -1)
        x_flat = torch.div(
            x_flat - torch.min(x_flat, dim=-1, keepdim=True)[0],
            torch.max(x_flat, dim=-1, keepdim=True)[0]
            - torch.min(x_flat, dim=-1, keepdim=True)[0],
        )
        x = x_flat.reshape(batch, ch, h, w)
        return x

    y_true = normalize_signal(y_true)
    y_pred = normalize_signal(y_pred)

    return ssim_criterion(y_true, y_pred)


def ssim_stand_with_same_minmax(y_true, y_pred, ssim_criterion, eps=1e-6):
    def normalize_signal_with_given_minmax(x, a, b):
        x = torch.div(x - a, b - a)
        return x

    a = torch.minimum(torch.min(y_true), torch.min(y_pred))
    b = torch.maximum(torch.max(y_true), torch.max(y_pred))
    y_true = normalize_signal_with_given_minmax(y_true, a, b)
    y_pred = normalize_signal_with_given_minmax(y_pred, a, b)

    return ssim_criterion(y_true, y_pred)
