"""
Copyright (c) 2025 Bahareh Tolooshams

finetune human reduced-time

:author: Bahareh Tolooshams
"""

import torch
from tqdm import tqdm
import configmypy
from datetime import datetime
from timeit import default_timer
import os
import json
import pickle

os.environ["WANDB_MODE"] = "run"

import datasetloader, lossfunc, utils


def init_params():
    pipe = configmypy.ConfigPipeline(
        [
            configmypy.YamlConfig(
                config_file="./human_finetune_training_reducedtime.yaml",
                config_name="default",
                config_folder="../config",
            ),
        ]
    )
    params = pipe.read_conf()

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


def main():

    # init parameters -------------------------------------------------------#
    params_init = init_params()

    ssim_alpha = params_init.ssim_alpha
    mse_alpha = params_init.mse_alpha
    mae_alpha = params_init.mae_alpha
    ssim_flag = params_init.ssim_flag
    ssim_win_size = params_init.ssim_win_size
    ssim_normalize_data_into_range = params_init.ssim_normalize_data_into_range
    ssim_normalize_together = params_init.ssim_normalize_together
    ssim_divide_by = params_init.ssim_divide_by

    params = pickle.load(
        open(os.path.join(params_init.model_path, "params.pickle"), "rb")
    )
    params.cudan = params_init.cudan
    params.res_path = params_init.model_path

    cudan = params.cudan
    device = torch.device(f"cuda:{cudan}" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}!")
    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    train_finetune_dataset_name = params_init.train_finetune_dataset.split("/")[-1]
    num_train_data = params_init.num_data
    model_name = params_init.model_path.split("/")[-1]
    params.out_path = f"{params_init.out_path}_{train_finetune_dataset_name}_{model_name}_{num_train_data}data_reducedtime_{random_date}"
    if not os.path.exists(params.out_path):
        os.makedirs(params.out_path)
    if not os.path.exists(os.path.join(params.out_path, "model")):
        os.makedirs(os.path.join(params.out_path, "model"))

    params.train_finetune_dataset = params_init.train_finetune_dataset
    params.svd_for_training = params_init.svd_for_training
    params.svd_for_training_limited = params_init.svd_for_training_limited
    params.num_frames_full = params_init.num_frames_full
    params.num_channels = params_init.num_channels
    params.loader.crop_dim = params_init.crop_dim
    params.loader.num_workers = params_init.num_workers
    params.num_data = params_init.num_data
    params.remove_top_pixels = params_init.remove_top_pixels

    if "ssim_divide_by" not in params.loss:
        params.loss.ssim_divide_by = 1

    params.opt.lr_finetune = params_init.lr_finetune
    params.opt.num_steps = params_init.num_steps

    print(f"number of frames: {params.num_frames}")

    # compute standardization constant --------------------------------------#
    (
        standardization_constant_folder,
        standardization_constant_name,
    ) = params.standardization_constant_path.split("/")[-2:]
    params.standardization_constant_path = os.path.join(
        "../data/standardization_constants",
        standardization_constant_folder,
        standardization_constant_name,
    )

    standardization_constant = utils.get_standardization_constant(params)

    # dump params  ---------------------------------------------------------#
    with open(os.path.join(params.out_path, "params.txt"), "w") as file:
        file.write(json.dumps(params, sort_keys=True, separators=("\n", ":")))
    with open(os.path.join(params.out_path, "params.pickle"), "wb") as file:
        pickle.dump(params, file)

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

    # create dataset and dataloader ------------------------------------------#
    if params.model == "VARSfUSI":
        train_dataset = (
            datasetloader.FUSdatasetOnlineSettingforTrainingDynamicTrainSizeIQDOPsvd(
                params.train_finetune_dataset,
                svd=params.svd_for_training,
                svd_for_limited=params.svd_for_training_limited,
                num_data=params.num_data,
                num_frames=params.num_frames,
                iq_signal_mode=params.iq_signal_mode,
                standardization_constant=standardization_constant,
                take_random_window=params.loader.take_random_window,
                transform=transform_train,
                num_frames_full=params.num_frames_full,
                num_channels=params.num_channels,
                total_frames=params.num_frames_full,
                remove_top_pixels=params.remove_top_pixels,
            )
        )
    else:
        train_dataset = (
            datasetloader.FUSdatasetOnlineSettingforTrainingDynamicTrainSize(
                params.train_finetune_dataset,
                svd=params.svd_for_training,
                num_data=params.num_data,
                num_frames=params.num_frames,
                iq_signal_mode=params.iq_signal_mode,
                standardization_constant=standardization_constant,
                take_random_window=params.loader.take_random_window,
                transform=transform_train,
                num_frames_full=params.num_frames_full,
                num_channels=params.num_channels,
                remove_top_pixels=params.remove_top_pixels,
            )
        )

    # create dataloaders -----------------------------------------#
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=params.loader.train_batch_size,
        num_workers=params.loader.num_workers,
    )

    # create model ---------------------------------------------------------#
    net, params = utils.get_model(params)
    net.to(device)

    # create optimizer ---------------------------------------#
    ckpt = torch.load(f"{params.res_path}/model/model_final.pt", map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=params.opt.lr_finetune,
        weight_decay=params.opt.weight_decay,
    )

    net.to(device)
    net.train()

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

    # train  ---------------------------------------------------------------#
    print("start training.")

    steps_ctr = 0
    stop_training = False

    start_time = default_timer()

    while not stop_training:

        net.train()

        for idx, data_tuple in tqdm(
            enumerate(train_loader), disable=params.disable_inner_loop
        ):

            steps_ctr += 1

            if steps_ctr % 10 == 0:
                print("steps_ctr", steps_ctr)

            if steps_ctr > params.opt.num_steps:
                stop_training = True
                break

            if params.model == "VARSfUSI":
                (iq_signal, dop_signal, dop_signal_with_svd, _) = data_tuple

            else:
                (iq_signal, dop_signal, _) = data_tuple

            iq_signal = iq_signal.to(device)
            dop_signal = dop_signal.to(device)

            dop_signal = torch.unsqueeze(dop_signal, dim=1)

            if params.model == "VARSfUSI":
                dop_signal_with_svd = dop_signal_with_svd.to(device)
                dop_signal_with_svd = torch.unsqueeze(dop_signal_with_svd, dim=1)

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

            save_model(net, optimizer, loss, params.out_path, steps_ctr, f"last")

    save_model(net, optimizer, loss, params.out_path, steps_ctr, f"final")

    print("training time:", (default_timer() - start_time) / 60, "min")


if __name__ == "__main__":
    main()
