"""
Copyright (c) 2025 Bahareh Tolooshams

finetune monkey reduced-time

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import os
import json
import pickle
import pytorch_msssim

os.environ["WANDB_MODE"] = "run"

import datasetloader, lossfunc, utils, model, metrics


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proj-name",
        type=str,
        help="proj name",
        default="fusi-finetune-monkey",
    )
    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/xxx",
    )
    parser.add_argument(
        "--num_train_finetune",
        type=int,
        help="train finetune dataset",
        default=60,
    )
    parser.add_argument("--num_steps", type=int, help="epochs fine tune", default=3600)
    parser.add_argument(
        "--train_finetune_dataset",
        type=str,
        help="train finetune dataset",
        default="/data/fus/S1/train",
    )
    parser.add_argument(
        "--val_finetune_dataset",
        type=str,
        help="val finetune dataset",
        default=None,
    )
    parser.add_argument(
        "--test_finetune_dataset",
        type=str,
        help="test finetine dataset",
        default=None,
    )
    parser.add_argument(
        "--test_out_finetune_dataset",
        type=str,
        help="test out finetine dataset",
        default=None,
    )
    parser.add_argument(
        "--crop-dim",
        type=list,
        help="crop_dim",
        default=[136, 136],
    )
    parser.add_argument("--svd", type=int, help="svd thresholding", default=38)
    parser.add_argument(
        "--svd_for_limited",
        type=int,
        help="svd thresholding for 32 limited frames",
        default=75,
    )
    parser.add_argument("--total_frames", type=int, help="total frames", default=250)

    parser.add_argument("--cudan", type=int, help="cuda number", default=0)
    parser.add_argument(
        "--lr-finetune", type=int, help="lr for fine tuning", default=1e-4
    )

    params = parser.parse_args()

    return params


def save_model(model, optimizer, loss, out_path, steps, name):
    model_path = os.path.join(out_path, "model", "model_{}.pt".format(name))
    torch.save(
        {
            "steps": steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        model_path,
    )
    return


def compute_loss(
    net, params, dataloader, criterion, standardization_constant, device="cpu"
):
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

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()

    test_period = 256

    ssim_alpha = 0.0
    mse_alpha = 0.5
    mae_alpha = 0.5
    ssim_flag = False
    ssim_win_size = 7
    ssim_normalize_data_into_range = False
    ssim_normalize_together = False
    ssim_divide_by = 1

    params = pickle.load(
        open(os.path.join(params_init.res_path, "params.pickle"), "rb")
    )
    params.cudan = params_init.cudan
    params.res_path = params_init.res_path

    params.proj_name = params_init.proj_name

    params.num_train_finetune = params_init.num_train_finetune
    params.train_finetune_dataset = params_init.train_finetune_dataset
    params.val_finetune_dataset = params_init.val_finetune_dataset
    params.test_finetune_dataset = params_init.test_finetune_dataset
    params.test_out_finetune_dataset = params_init.test_out_finetune_dataset

    session_name = params_init.train_finetune_dataset.split("/")[-2]
    train_name = f"reducedtime_{params_init.num_train_finetune}_{session_name}"

    if "ssim_divide_by" not in params.loss:
        params.loss.ssim_divide_by = 1

    params.log_info_step_period = 10
    params.log_fig_step_period = 1000

    params.opt.lr_finetune = params_init.lr_finetune
    params.opt.num_steps = params_init.num_steps

    params.svd = params_init.svd

    params.svd_for_limited = params_init.svd_for_limited

    print("project: {}".format(params.proj_name))
    print("exp: {}".format(params.exp_name))

    cudan = params.cudan
    device = torch.device(f"cuda:{cudan}" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}!")
    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    print(f"number of frames: {params.num_frames}")

    if params.wandb:
        wandb.init(
            dir="/tmp",
            entity="btolooshams",
            project=params.proj_name,
            group="{}_{}".format(params.exp_name, train_name),
            name="{}_{}_{}".format(params.exp_name, random_date, train_name),
            id="{}_{}_{}_{}".format(
                params.proj_name,
                params.exp_name,
                random_date,
                train_name,
            ),
        )

    params.out_path = "{}_{}_{}".format(params.res_path, train_name, random_date)
    if not os.path.exists(params.out_path):
        os.makedirs(params.out_path)
    if not os.path.exists(os.path.join(params.out_path, "model")):
        os.makedirs(os.path.join(params.out_path, "model"))

    out_path = params.out_path

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
                    params_init.crop_dim,
                    scale=params.loader.train_random_crop_resize.scale,
                    ratio=params.loader.train_random_crop_resize.ratio,
                    antialias=params.loader.train_random_crop_resize.antialias,
                ),
            ]
        else:
            transform_train = [
                utils.FUSRandomCrop(params_init.crop_dim, pad_if_needed=True),
            ]
    else:
        transform_train = [
            utils.FUSCenterCrop(params_init.crop_dim),
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
        utils.FUSCenterCrop(params_init.crop_dim),
    ]
    transform_test = [
        utils.FUSCenterCrop(params_init.crop_dim),
    ]
    transform_test_out = [
        utils.FUSCenterCrop(params_init.crop_dim),
    ]

    # create dataset and dataloader ------------------------------------------#
    if params.model == "VARSfUSI":
        train_dataset = datasetloader.FUSdatasetDynamicsTrainSizeIQDOPsvd(
            params.train_finetune_dataset,
            num_data=params.num_train_finetune,
            svd=params.svd_for_limited,
            num_frames=params.num_frames,
            iq_signal_mode=params.iq_signal_mode,
            standardization_constant=standardization_constant,
            take_random_window=params.loader.take_random_window,
            transform=transform_train,
            total_frames=params_init.total_frames,
        )
        if params.val_finetune_dataset:
            val_dataset = datasetloader.FUSdatasetDynamicsTrainSizeIQDOPsvd(
                params.val_finetune_dataset,
                num_data=None,
                svd=params.svd_for_limited,
                num_frames=params.num_frames,
                iq_signal_mode=params.iq_signal_mode,
                standardization_constant=standardization_constant,
                take_random_window=False,
                transform=transform_val,
                total_frames=params_init.total_frames,
            )
        else:
            val_dataset = None
        if params.test_finetune_dataset:
            test_dataset = datasetloader.FUSdatasetDynamicsTrainSizeIQDOPsvd(
                params.test_finetune_dataset,
                num_data=None,
                svd=params.svd_for_limited,
                num_frames=params.num_frames,
                iq_signal_mode=params.iq_signal_mode,
                standardization_constant=standardization_constant,
                take_random_window=False,
                transform=transform_test,
                total_frames=params_init.total_frames,
            )
        else:
            test_dataset = None
        if params.test_out_finetune_dataset:
            test_dataset_out = datasetloader.FUSdatasetDynamicsTrainSizeIQDOPsvd(
                params.test_out_finetune_dataset,
                num_data=None,
                svd=params.svd_for_limited,
                num_frames=params.num_frames,
                iq_signal_mode=params.iq_signal_mode,
                standardization_constant=standardization_constant,
                take_random_window=False,
                transform=transform_test_out,
                total_frames=params_init.total_frames,
            )
        else:
            test_dataset_out = None
    else:
        train_dataset = datasetloader.FUSdatasetDynamicsTrainSize(
            params.train_finetune_dataset,
            num_data=params.num_train_finetune,
            num_frames=params.num_frames,
            iq_signal_mode=params.iq_signal_mode,
            standardization_constant=standardization_constant,
            take_random_window=params.loader.take_random_window,
            transform=transform_train,
        )
        if params.val_finetune_dataset:
            val_dataset = datasetloader.FUSdatasetDynamicsTrainSize(
                params.val_finetune_dataset,
                num_data=None,
                num_frames=params.num_frames,
                iq_signal_mode=params.iq_signal_mode,
                standardization_constant=standardization_constant,
                take_random_window=False,
                transform=transform_val,
            )
        else:
            val_dataset = None
        if params.test_finetune_dataset:
            test_dataset = datasetloader.FUSdatasetDynamicsTrainSize(
                params.test_finetune_dataset,
                num_data=None,
                num_frames=params.num_frames,
                iq_signal_mode=params.iq_signal_mode,
                standardization_constant=standardization_constant,
                take_random_window=False,
                transform=transform_test,
            )
        else:
            test_dataset = None
        if params.test_out_finetune_dataset:
            test_dataset_out = datasetloader.FUSdatasetDynamicsTrainSize(
                params.test_out_finetune_dataset,
                num_data=None,
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
    net, params = utils.get_model(params)
    net.to(device)

    # create optimizer and scheduler ---------------------------------------#
    print("create optimizer and scheduler for training.")

    ckpt = torch.load(f"{params.res_path}/model/model_final.pt", map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=params.opt.lr_finetune,
        weight_decay=params.opt.weight_decay,
    )

    net.to(device)
    net.train()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,
        gamma=1,
    )

    # create loss criterion  ------------------------------------------------#
    criterion = lossfunc.FUSLoss(
        ssim_alpha=ssim_alpha,
        mse_alpha=mse_alpha,
        mae_alpha=mae_alpha,
        ssim_flag=ssim_flag,
        ssim_win_size=ssim_win_size,
        ssim_normalize_data_into_range=ssim_normalize_data_into_range,
        ssim_normalize_together=ssim_normalize_together,
        ssim_divide_by=ssim_divide_by,
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

    steps_ctr = 0
    stop_training = False

    while not stop_training:

        net.train()

        for idx, loaded_data in tqdm(
            enumerate(train_loader), disable=params.disable_inner_loop
        ):

            if params.model == "VARSfUSI":
                (iq_signal, dop_signal, dop_signal_with_svd, _) = loaded_data
                dop_signal_with_svd = dop_signal_with_svd.to(device)
            else:
                (iq_signal, dop_signal, _) = loaded_data

            steps_ctr += 1

            if steps_ctr > params.opt.num_steps:
                stop_training = True
                break

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

            # backward to update kernels
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # gradient clipping
            if params.opt.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), max_norm=params.opt.grad_clip
                )

            optimizer.step()

            total_loss = loss.item()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.item())
            else:
                scheduler.step()

            if (steps_ctr + 1) % params.log_info_step_period == 0:
                if (steps_ctr + 1) % (params.log_info_step_period * test_period) == 0:
                    (
                        test_loss,
                        test_nmae,
                        test_nmse,
                        test_psnr,
                        test_ssim,
                    ) = compute_loss(
                        net,
                        params,
                        test_loader,
                        criterion,
                        standardization_constant,
                        device,
                    )
                    (
                        test_loss_out,
                        test_nmae_out,
                        test_nmse_out,
                        test_psnr_out,
                        test_ssim_out,
                    ) = compute_loss(
                        net,
                        params,
                        test_loader_out,
                        criterion,
                        standardization_constant,
                        device,
                    )

                if val_loader:
                    val_loss, val_nmae, val_nmse, val_psnr, val_ssim = compute_loss(
                        net,
                        params,
                        val_loader,
                        criterion,
                        standardization_constant,
                        device,
                    )

                if (steps_ctr + 1) % (params.log_info_step_period * test_period) == 0:
                    if test_loss <= best_test_loss:
                        print(f"{steps_ctr}: Test loss is improved!")
                        best_test_loss = test_loss

                    if test_loss_out <= best_test_loss_out:
                        print(f"{steps_ctr}: Test out loss is improved!")
                        best_test_loss_out = test_loss_out

                if val_loader:
                    if val_loss <= best_val_loss:
                        print(f"{steps_ctr}: Val loss is improved!")
                        best_val_loss = val_loss
                        save_model(
                            net, optimizer, loss, out_path, steps_ctr, "best_val"
                        )

                # for training
                curr_nmae = metrics.nmae(dop_signal, dop_signal_est).item()
                curr_nmse = metrics.nmse(dop_signal, dop_signal_est).item()
                curr_psnr = metrics.psnr_stand_with_same_minmax(
                    dop_signal, dop_signal_est
                ).item()
                curr_ssim = metrics.ssim_stand_with_same_minmax(
                    dop_signal, dop_signal_est, ssim_criterion
                ).item()

                total_nmae = curr_nmae
                total_nmse = curr_nmse
                total_psnr = curr_psnr
                total_ssim = curr_ssim

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
                        step=steps_ctr + 1,
                    )

                    if (steps_ctr + 1) % (
                        params.log_info_step_period * test_period
                    ) == 0:
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
                                step=steps_ctr + 1,
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
                                step=steps_ctr + 1,
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
                            step=steps_ctr + 1,
                        )

            if params.wandb.flag and (steps_ctr + 1) % params.log_fig_step_period == 0:
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
                wandb.log({"dop": fig}, step=steps_ctr + 1)
                plt.close()

            save_model(net, optimizer, loss, out_path, steps_ctr, "last")

    save_model(net, optimizer, loss, out_path, steps_ctr, "final")


if __name__ == "__main__":
    main()
