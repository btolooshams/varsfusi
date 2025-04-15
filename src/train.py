"""
Copyright (c) 2025 Bahareh Tolooshams

train script

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
from tqdm import tqdm
import wandb
import configmypy
from datetime import datetime
from timeit import default_timer
import matplotlib.pyplot as plt
import argparse
import os
import json
import pickle
import pytorch_msssim

import datasetloader, lossfunc, model, utils, metrics


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-filename",
        type=str,
        help="config filename",
        default="./varsfusi.yaml",
        # default="./varsfusi_wo_sg.yaml",
        # default="./varsfusi_real_wo_sg.yaml",
        # default="./deepfus.yaml",
        # default="./deepfus_base.yaml",
        # default="./dncnn.yaml",
    )
    params = parser.parse_args()

    return params


def save_model(model, optimizer, loss, out_path, epoch, name):
    model_path = os.path.join(out_path, "model", "model_{}.pt".format(name))
    print(f"saving model to {model_path}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        model_path,
    )
    return


def compute_loss(net, dataloader, criterion, params, device="cpu"):
    net.eval()

    ssim_criterion = pytorch_msssim.SSIM(
        data_range=1,
        size_average=True,
        channel=1,
        win_size=7,
    )

    if dataloader:
        with torch.no_grad():
            total_loss = 0.0
            nmae = 0.0
            nmse = 0.0
            psnr_stand_with_same_minmax = 0.0
            ssim_stand_with_same_minmax = 0.0
            batch_ctr = 0

            for idx, loaded_data in tqdm(enumerate(dataloader), disable=True):

                if params.model == "VARSfUSI":
                    (iq_signal, dop_signal, dop_signal_with_svd, _) = loaded_data
                    dop_signal_with_svd = dop_signal_with_svd.to(device)
                    dop_signal_with_svd = torch.unsqueeze(dop_signal_with_svd, dim=1)
                else:
                    (iq_signal, dop_signal, _) = loaded_data

                batch_ctr += 1

                iq_signal = iq_signal.to(device)
                dop_signal = dop_signal.to(device)

                dop_signal = torch.unsqueeze(dop_signal, dim=1)

                # forward encoder
                if params.model == "VARSfUSI":
                    dop_signal_est = net(iq_signal, dop_signal_with_svd)
                else:
                    dop_signal_est = net(iq_signal)

                # compute loss
                loss = criterion(dop_signal, dop_signal_est)

                total_loss += loss.item()

                nmae += metrics.nmae(dop_signal, dop_signal_est).item()
                nmse += metrics.nmse(dop_signal, dop_signal_est).item()
                psnr_stand_with_same_minmax += metrics.psnr_stand_with_same_minmax(
                    dop_signal, dop_signal_est
                ).item()
                ssim_stand_with_same_minmax += metrics.ssim_stand_with_same_minmax(
                    dop_signal, dop_signal_est, ssim_criterion
                ).item()

        total_loss /= batch_ctr
        nmae /= batch_ctr
        nmse /= batch_ctr
        psnr_stand_with_same_minmax /= batch_ctr
        ssim_stand_with_same_minmax /= batch_ctr

    else:
        (
            total_loss,
            nmae,
            nmse,
            psnr_stand_with_same_minmax,
            ssim_stand_with_same_minmax,
        ) = (0, 0, 0, 0, 0)

    net.train()

    return (
        total_loss,
        nmae,
        nmse,
        psnr_stand_with_same_minmax,
        ssim_stand_with_same_minmax,
    )


def main():
    print("Train on fUS dataset.")

    ctr_for_train_print = 50
    test_period = 10

    # init parameters -------------------------------------------------------#
    params_init = init_params()

    pipe = configmypy.ConfigPipeline(
        [
            configmypy.YamlConfig(
                params_init.config_filename,
                config_name="default",
                config_folder="../config",
            ),
            configmypy.ArgparseConfig(
                infer_types=True, config_name=None, config_file=None
            ),
            configmypy.YamlConfig(config_folder="../config"),
        ]
    )
    params = pipe.read_conf()

    print("project: {}".format(params.proj_name))
    print("exp: {}".format(params.exp_name))

    if params.data_path.train.split("/")[1] == "central":
        params.loader.num_workers = 1
        print(f"set num_workers for hpc to {params.loader.num_workers}.")

    cudan = params.cudan
    device = torch.device(f"cuda:{cudan}" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}!")
    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    print(f"number of frames: {params.num_frames}")
    if params.summary_test:
        params.wandb.flag = False

    if params.wandb.flag:
        wandb.init(
            dir=params.wandb.dir,
            entity=params.wandb.entity,
            project=params.proj_name,
            group=params.exp_name,
            name="{}_{}".format(params.exp_name, random_date),
            id="{}_{}_{}".format(params.proj_name, params.exp_name, random_date),
        )

    # create folder for results ---------------------------------------------#
    out_path = os.path.join(
        "..", "results", "{}_{}".format(params.exp_name, random_date)
    )
    params.out_path = out_path
    if not os.path.exists(params.out_path):
        os.makedirs(params.out_path)
    if not os.path.exists(os.path.join(params.out_path, "model")):
        os.makedirs(os.path.join(params.out_path, "model"))

    # dump params  ---------------------------------------------------------#
    with open(os.path.join(params.out_path, "params.txt"), "w") as file:
        file.write(json.dumps(params, sort_keys=True, separators=("\n", ":")))
    with open(os.path.join(params.out_path, "params.pickle"), "wb") as file:
        pickle.dump(params, file)

    # compute standardization constant --------------------------------------#
    print("loading standardization_constant")
    standardization_constant = utils.get_standardization_constant(params)

    # create transforms for dataset -----------------------------------------#
    if params.loader.train_random_crop:
        if params.loader.train_random_crop_resize.flag:
            transform_train = [
                utils.FUSRandomResizedCrop(
                    params.loader.crop_dim,
                    scale=params.loader.train_random_crop_resize.scale,
                    ratio=params.loader.train_random_crop_resize.ratio,
                    antialias=params.loader.train_random_crop_resize.antialias,
                ),
            ]
        else:
            transform_train = [
                utils.FUSRandomCrop(params.loader.crop_dim, pad_if_needed=True),
            ]
    else:
        transform_train = [
            utils.FUSCenterCrop(params.loader.crop_dim),
        ]

    if params.loader.train_rotate.flag:
        transform_train.append(
            utils.FUSRandomRotation(degrees=params.loader.train_rotate.degrees)
        )

    if params.loader.train_flip_h:
        transform_flip_h = utils.FUSRandomHorizontalFlip(p=0.5)
        transform_train.append(transform_flip_h)

    if params.loader.train_flip_v:
        transform_flip_v = utils.FUSRandomVerticalFlip(p=0.5)
        transform_train.append(transform_flip_v)

    if params.loader.train_noise.flag:
        transform_add_noise = utils.FUSAddGaussianNoise(
            p=params.loader.train_noise.p,
            mean=params.loader.train_noise.mean,
            std=params.loader.train_noise.std,
        )
        transform_train.append(transform_add_noise)

    if params.loader.train_masking.flag:
        transform_train.append(
            utils.FUSRandomMasking(
                p=params.loader.train_masking.p,
                mask_prob_range=params.loader.train_masking.mask_prob_range,
            )
        )

    transform_val = [
        utils.FUSCenterCrop(params.loader.crop_dim),
    ]
    transform_test = [
        utils.FUSCenterCrop(params.loader.crop_dim),
    ]
    transform_test_out = [
        utils.FUSCenterCrop(params.loader.crop_dim),
    ]

    # create dataset and dataloader ------------------------------------------#
    train_dataset = getattr(datasetloader, params.loader.function)(
        params.data_path.train,
        num_frames=params.num_frames,
        iq_signal_mode=params.iq_signal_mode,
        standardization_constant=standardization_constant,
        take_random_window=params.loader.take_random_window,
        transform=transform_train,
    )
    if params.data_path.val:
        val_dataset = getattr(datasetloader, params.loader.function)(
            params.data_path.val,
            num_frames=params.num_frames,
            iq_signal_mode=params.iq_signal_mode,
            standardization_constant=standardization_constant,
            take_random_window=False,
            transform=transform_val,
        )
    else:
        val_dataset = None
    if params.data_path.test:
        test_dataset = getattr(datasetloader, params.loader.function)(
            params.data_path.test,
            num_frames=params.num_frames,
            iq_signal_mode=params.iq_signal_mode,
            standardization_constant=standardization_constant,
            take_random_window=False,
            transform=transform_test,
        )
    else:
        test_dataset = None
    if params.data_path.test_out:
        test_dataset_out = getattr(datasetloader, params.loader.function)(
            params.data_path.test_out,
            num_frames=params.num_frames,
            iq_signal_mode=params.iq_signal_mode,
            standardization_constant=standardization_constant,
            take_random_window=False,
            transform=transform_test_out,
        )
    else:
        test_dataset_out = None
    num_train = len(train_dataset)
    if val_dataset:
        num_val = len(val_dataset)
    else:
        num_val = 0

    if test_dataset:
        num_test = len(test_dataset)
    else:
        num_test = 0
    if test_dataset_out:
        num_test_out = len(test_dataset_out)
    else:
        num_test_out = 0

    print(
        f"there are {num_train}/{num_val}/{num_test}/{num_test_out} train/val/test/test_out data."
    )

    # create dataloaders -----------------------------------------#
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=params.loader.train_batch_size,
        num_workers=params.loader.num_workers,
    )
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=params.loader.test_batch_size,
            num_workers=params.loader.num_workers,
        )
    else:
        val_loader = None
    if test_dataset:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=params.loader.test_batch_size,
            num_workers=params.loader.num_workers,
        )
    else:
        test_loader = None
    if test_dataset_out:
        test_loader_out = torch.utils.data.DataLoader(
            test_dataset_out,
            shuffle=False,
            batch_size=params.loader.test_batch_size,
            num_workers=params.loader.num_workers,
        )
    else:
        test_loader_out = None

    # create model ---------------------------------------------------------#
    print("create model.")
    net = getattr(model, params.model)(params)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("num_params:", num_params)

    if params.summary_test:
        exit()

    net.to(device)

    # create optimizer and scheduler ---------------------------------------#
    print("create optimizer and scheduler for training.")
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=params.opt.lr,
        weight_decay=params.opt.weight_decay,
    )

    if params.opt.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=params.opt.scheduler_gamma,
            patience=params.opt.scheduler_patience,
            mode="min",
        )
    elif params.opt.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params.opt.scheduler_tmax
        )
    elif params.opt.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.opt.scheduler_stepsize,
            gamma=params.opt.scheduler_gamma,
        )
    else:
        raise ValueError(f"Scheduler name is not implemented!")

    # create loss criterion  ------------------------------------------------#
    criterion = lossfunc.FUSLoss(
        ssim_alpha=params.loss.ssim_alpha,
        mse_alpha=params.loss.mse_alpha,
        mae_alpha=params.loss.mae_alpha,
        ssim_flag=params.loss.ssim_flag,
        ssim_win_size=params.loss.ssim_win_size,
        ssim_normalize_data_into_range=params.loss.ssim_normalize_data_into_range,
        ssim_normalize_together=params.loss.ssim_normalize_together,
        ssim_divide_by=params.loss.ssim_divide_by,
    )

    best_test_loss_out = float("inf")
    best_test_loss = float("inf")
    best_val_loss = float("inf")

    ssim_criterion = pytorch_msssim.SSIM(
        data_range=1,
        size_average=True,
        channel=1,
        win_size=7,
    )

    # train  ---------------------------------------------------------------#
    print("start training.")

    for epoch in tqdm(range(params.opt.num_epochs), disable=params.tqdm_prints_disable):
        net.train()
        total_loss = 0.0
        total_nmae = 0.0
        total_nmse = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        t1 = default_timer()

        ctr_for_train = 0
        batch_ctr = 0

        for idx, loaded_data in tqdm(
            enumerate(train_loader), disable=params.disable_inner_loop
        ):

            if params.model == "VARSfUSI":
                (iq_signal, dop_signal, dop_signal_with_svd, _) = loaded_data
                dop_signal_with_svd = dop_signal_with_svd.to(device)
                dop_signal_with_svd = torch.unsqueeze(dop_signal_with_svd, dim=1)
            else:
                (iq_signal, dop_signal, _) = loaded_data

            batch_ctr += 1
            ctr_for_train += 1

            iq_signal = iq_signal.to(device)
            dop_signal = dop_signal.to(device)

            dop_signal = torch.unsqueeze(dop_signal, dim=1)

            ################ this is for mixing
            if (
                params.loader.train_mix.flag
                and torch.rand(1) < params.loader.train_mix.p
            ):
                mixing = np.random.beta(0.2, 0.2)
                first_batch = int(iq_signal.shape[0] / 2)

                iq_signal = (
                    mixing * iq_signal[:first_batch]
                    + (1.0 - mixing) * iq_signal[first_batch:]
                )
                dop_signal = (
                    mixing * dop_signal[:first_batch]
                    + (1.0 - mixing) * dop_signal[first_batch:]
                )

                if params.model == "VARSfUSI":
                    dop_signal_with_svd = (
                        mixing * dop_signal_with_svd[:first_batch]
                        + (1.0 - mixing) * dop_signal_with_svd[first_batch:]
                    )
            ################

            # forward encoder
            if params.model == "VARSfUSI":
                dop_signal_est = net(iq_signal, dop_signal_with_svd)
            else:
                dop_signal_est = net(iq_signal)

            # compute loss
            loss = criterion(dop_signal, dop_signal_est)

            # backward to update kernels
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # gradient clipping
            if params.opt.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), max_norm=params.opt.grad_clip
                )

            optimizer.step()

            total_loss += loss.item()

            curr_nmae = metrics.nmae(dop_signal, dop_signal_est).item()
            curr_nmse = metrics.nmse(dop_signal, dop_signal_est).item()
            curr_psnr = metrics.psnr_stand_with_same_minmax(
                dop_signal, dop_signal_est
            ).item()
            curr_ssim = metrics.ssim_stand_with_same_minmax(
                dop_signal, dop_signal_est, ssim_criterion
            ).item()

            total_nmae += curr_nmae
            total_nmse += curr_nmse
            total_psnr += curr_psnr
            total_ssim += curr_ssim

            if ctr_for_train < ctr_for_train_print:
                if (ctr_for_train) % 10 == 0:
                    print(
                        "train",
                        curr_psnr,
                        curr_ssim,
                        curr_nmse,
                    )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(total_loss)
        else:
            scheduler.step()

        total_loss /= batch_ctr
        total_nmae /= batch_ctr
        total_nmse /= batch_ctr
        total_psnr /= batch_ctr
        total_ssim /= batch_ctr

        if (epoch + 1) % params.log_info_epoch_period == 0:
            if (epoch + 1) % (params.log_info_epoch_period * test_period) == 0:
                test_loss, test_nmae, test_nmse, test_psnr, test_ssim = compute_loss(
                    net, test_loader, criterion, params, device
                )
                (
                    test_loss_out,
                    test_nmae_out,
                    test_nmse_out,
                    test_psnr_out,
                    test_ssim_out,
                ) = compute_loss(net, test_loader_out, criterion, params, device)
            val_loss, val_nmae, val_nmse, val_psnr, val_ssim = compute_loss(
                net, val_loader, criterion, params, device
            )

            print(
                f"epoch {epoch}: train loss {total_loss:.4f}, psnr {total_psnr:.2f}, nmse {total_nmse:.4f}, ssim {total_ssim:.4f}"
            )
            if (epoch + 1) % (params.log_info_epoch_period * test_period) == 0:
                print(
                    f"epoch {epoch}: test loss {test_loss:.4f}, psnr {test_psnr:.2f}, nmse {test_nmse:.4f}, ssim {test_ssim:.4f}"
                )
                print(
                    f"epoch {epoch}: test out loss {test_loss_out:.4f}, psnr {test_psnr_out:.2f}, nmse {test_nmse_out:.4f}, ssim {test_ssim_out:.4f}"
                )
            print(
                f"epoch {epoch}: val loss {val_loss:.4f}, psnr {val_psnr:.2f}, nmse {val_nmse:.4f}, ssim {val_ssim:.4f}"
            )

            if (epoch + 1) % (params.log_info_epoch_period * test_period) == 0:
                if test_loss <= best_test_loss:
                    print("Test loss is improved!")
                    best_test_loss = test_loss

                if test_loss_out <= best_test_loss_out:
                    print("Test out loss is improved!")
                    best_test_loss_out = test_loss_out

            if val_loss <= best_val_loss:
                print("Val loss is improved!")
                best_val_loss = val_loss
                save_model(net, optimizer, loss, out_path, epoch, "best_val")

            # log info
            if params.wandb.flag:
                wandb.log(
                    {
                        "train loss": total_loss,
                        "train nmae": total_nmae,
                        "train nmse": total_nmse,
                        "train psnr": total_psnr,
                        "train ssim": total_ssim,
                    },
                    step=epoch + 1,
                )

                if (epoch + 1) % (params.log_info_epoch_period * test_period) == 0:
                    if test_loader:
                        # log test info
                        wandb.log(
                            {
                                "test loss": test_loss,
                                "test nmae": test_nmae,
                                "test nmse": test_nmse,
                                "test psnr": test_psnr,
                                "test ssim": test_ssim,
                            },
                            step=epoch + 1,
                        )

                    if test_loader_out:
                        # log test out info
                        wandb.log(
                            {
                                "test out loss": test_loss_out,
                                "test out nmae": test_nmae_out,
                                "test out nmse": test_nmse_out,
                                "test out psnr": test_psnr_out,
                                "test out ssim": test_ssim_out,
                            },
                            step=epoch + 1,
                        )

                if val_loader:
                    # log val info
                    wandb.log(
                        {
                            "val loss": val_loss,
                            "val nmae": val_nmae,
                            "val nmse": val_nmse,
                            "val psnr": val_psnr,
                            "val ssim": val_ssim,
                        },
                        step=epoch + 1,
                    )

        if params.wandb.flag and (epoch + 1) % params.log_fig_epoch_period == 0:
            # log fig
            dop = utils.unstandardize_general(
                dop_signal[0, 0],
                standardization_constant["dop_mean"],
                standardization_constant["dop_std"],
            )
            dop_est = utils.unstandardize_general(
                dop_signal_est[0, 0],
                standardization_constant["dop_mean"],
                standardization_constant["dop_std"],
            )

            a = np.minimum(
                torch.min(dop).clone().detach().cpu().numpy(),
                torch.min(dop_est).clone().detach().cpu().numpy(),
            )
            b = np.maximum(
                torch.max(dop).clone().detach().cpu().numpy(),
                torch.max(dop_est).clone().detach().cpu().numpy(),
            )

            fig = plt.figure(figsize=(3, 2))
            plt.subplot(121), plt.imshow(
                dop.clone().detach().cpu().numpy(),
                cmap="hot",
                vmin=a,
                vmax=b,
            )
            plt.title("desired"), plt.axis("off")
            plt.subplot(122), plt.imshow(
                dop_est.clone().detach().cpu().numpy(),
                cmap="hot",
                vmin=a,
                vmax=b,
            )
            plt.title("est"), plt.axis("off")
            wandb.log({"dop": fig}, step=epoch + 1)
            plt.close()

        epoch_train_time = default_timer() - t1
        print("epoch train time is {} min.".format(epoch_train_time / 60))

        save_model(net, optimizer, loss, out_path, epoch, "last")

    save_model(net, optimizer, loss, out_path, epoch, "final")

    return best_val_loss


if __name__ == "__main__":
    main()
