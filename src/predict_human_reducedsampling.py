"""
Copyright (c) 2025 Bahareh Tolooshams

predict human reduced-sampling

:author: Bahareh Tolooshams
"""

import numpy as np
import torch
import torchvision
import configmypy
import os
import pickle
import pytorch_msssim

import datasetloader, utils, metrics

cudan = 0
device = torch.device(f"cuda:{cudan}" if torch.cuda.is_available() else "cpu")


def init_params():
    pipe = configmypy.ConfigPipeline(
        [
            configmypy.YamlConfig(
                config_file="./human_finetuned_prediction_reducedsampling.yaml",
                config_name="default",
                config_folder="../config",
            ),
        ]
    )
    params = pipe.read_conf()

    return params


def compute_loss(
    dataset_for_svd,
    dataset_for_svd_full_frames,
    dataset_for_method,
    net,
    standardization_constant,
    params,
):

    ssim_criterion = pytorch_msssim.SSIM(
        data_range=1,
        size_average=True,
        channel=1,
        win_size=7,
    )

    res_method = {
        "nmae": list(),
        "nmse": list(),
        "psnr_stand_with_same_minmax": list(),
        "ssim_stand_with_same_minmax": list(),
    }
    res_oracle_svd = {
        "nmae": list(),
        "nmse": list(),
        "psnr_stand_with_same_minmax": list(),
        "ssim_stand_with_same_minmax": list(),
    }
    res_svd_fusi = {
        "nmae": list(),
        "nmse": list(),
        "psnr_stand_with_same_minmax": list(),
        "ssim_stand_with_same_minmax": list(),
    }

    with torch.no_grad():
        for ctr in range(params.start_bin, params.end_bin + 1):

            bin_num = f"{ctr}".zfill(3)
            fusblock_name = f"fUS_block_{bin_num}.bin"
            fusblock_path = os.path.join(params.test_data_path, fusblock_name)
            while not os.path.exists(fusblock_path):
                continue

            print(f"bin number: {bin_num}")

            # data_path_list contains the list of all bin data
            if dataset_for_svd is not None:
                dataset_for_svd.data_path_list = [fusblock_path]
                dataset_for_svd.num_data = 1

            if dataset_for_svd_full_frames is not None:
                dataset_for_svd_full_frames.data_path_list = [fusblock_path]
                dataset_for_svd_full_frames.num_data = 1

            dataset_for_method.data_path_list = [fusblock_path]
            dataset_for_method.num_data = 1

            # get from dataset_for_svd
            if dataset_for_svd is not None:
                _, dop_signal_svd, curr_input_path_svd = dataset_for_svd[0]
                dop_signal_svd = dop_signal_svd.to(device)
                dop_signal_svd = utils.standardize_general(
                    dop_signal_svd,
                    standardization_constant["dop_mean"],
                    standardization_constant["dop_std"],
                )

            # get from dataset_for_svd
            if dataset_for_svd_full_frames is not None:
                (
                    _,
                    dop_signal_svd_full_frames,
                    curr_input_path_svd_full_frames,
                ) = dataset_for_svd_full_frames[0]
                dop_signal_svd_full_frames = utils.standardize_general(
                    dop_signal_svd_full_frames,
                    standardization_constant["dop_mean"],
                    standardization_constant["dop_std"],
                )
                dop_signal_svd_full_frames = dop_signal_svd_full_frames.to(device)

            # get lower limit svd
            svd_org = params.svd_with_respect_to_full_frames
            iq_signal_for_ll = dataset_for_svd_full_frames[0][0]
            z_dim, x_dim, t_dim = iq_signal_for_ll.shape
            U, Λ, V = np.linalg.svd(
                iq_signal_for_ll.reshape(z_dim * x_dim, t_dim),
                full_matrices=False,
            )
            iqf_full = U[:, svd_org:] @ np.diag(Λ[svd_org:]) @ V.T[svd_org:]
            iqf_lower_limit = iqf_full[:, :: params.interleaved]
            iqf_lower_limit = iqf_lower_limit[:, : params.num_frames]
            dop_signal_svd_lowerlimit = np.mean(
                np.abs(iqf_lower_limit) ** 2, axis=-1
            ).reshape(z_dim, x_dim)
            dop_signal_svd_lowerlimit = torch.tensor(dop_signal_svd_lowerlimit)
            dop_signal_svd_lowerlimit = torch.tensor(dop_signal_svd_lowerlimit).to(
                device
            )
            dop_signal_svd_lowerlimit = utils.standardize_general(
                dop_signal_svd_lowerlimit,
                standardization_constant["dop_mean"],
                standardization_constant["dop_std"],
            )

            # get from dataset_for_method
            if params.model == "VARSfUSI":
                (
                    iq_signal,
                    dop_signal_limited_frames,
                    _,
                ) = dataset_for_method[0]
                iq_signal = iq_signal.to(device)
                dop_signal_limited_frames = dop_signal_limited_frames.to(device)
                iq_signal = torch.unsqueeze(iq_signal, dim=0)
                dop_signal_limited_frames = torch.unsqueeze(
                    dop_signal_limited_frames, dim=0
                )
                dop_signal_no = net(iq_signal, dop_signal_limited_frames)
            else:
                (
                    iq_signal,
                    _,
                ) = dataset_for_method[0]
                iq_signal = iq_signal.to(device)
                iq_signal = torch.unsqueeze(iq_signal, dim=0)
                dop_signal_no = net(iq_signal)
            dop_signal_no = torch.squeeze(dop_signal_no)
            dop_signal_no = dop_signal_no.detach()

            ######
            dop_signal_no = dop_signal_no[5:-6]

            svd_limited_nmse = metrics.nmse(
                dop_signal_svd_full_frames.unsqueeze(0).unsqueeze(1),
                dop_signal_svd.unsqueeze(0).unsqueeze(1),
            ).item()
            svd_limited_psnr = metrics.psnr_stand_with_same_minmax(
                dop_signal_svd_full_frames.unsqueeze(0).unsqueeze(1),
                dop_signal_svd.unsqueeze(0).unsqueeze(1),
            ).item()
            svd_limited_ssim = metrics.ssim_stand_with_same_minmax(
                dop_signal_svd_full_frames.unsqueeze(0).unsqueeze(1),
                dop_signal_svd.unsqueeze(0).unsqueeze(1),
                ssim_criterion,
            ).item()

            no_nmse = metrics.nmse(
                dop_signal_svd_full_frames.unsqueeze(0).unsqueeze(1),
                dop_signal_no.unsqueeze(0).unsqueeze(1),
            ).item()
            no_psnr = metrics.psnr_stand_with_same_minmax(
                dop_signal_svd_full_frames.unsqueeze(0).unsqueeze(1),
                dop_signal_no.unsqueeze(0).unsqueeze(1),
            ).item()
            no_ssim = metrics.ssim_stand_with_same_minmax(
                dop_signal_svd_full_frames.unsqueeze(0).unsqueeze(1),
                dop_signal_no.unsqueeze(0).unsqueeze(1),
                ssim_criterion,
            ).item()

            oracle_nmse = metrics.nmse(
                dop_signal_svd_full_frames.unsqueeze(0).unsqueeze(1),
                dop_signal_svd_lowerlimit.unsqueeze(0).unsqueeze(1),
            ).item()
            oracle_psnr = metrics.psnr_stand_with_same_minmax(
                dop_signal_svd_full_frames.unsqueeze(0).unsqueeze(1),
                dop_signal_svd_lowerlimit.unsqueeze(0).unsqueeze(1),
            ).item()
            oracle_ssim = metrics.ssim_stand_with_same_minmax(
                dop_signal_svd_full_frames.unsqueeze(0).unsqueeze(1),
                dop_signal_svd_lowerlimit.unsqueeze(0).unsqueeze(1),
                ssim_criterion,
            ).item()

            res_method["nmse"].append(no_nmse)
            res_method["psnr_stand_with_same_minmax"].append(no_psnr)
            res_method["ssim_stand_with_same_minmax"].append(no_ssim)

            # print(no_psnr, no_ssim, no_nmse)

            res_oracle_svd["nmse"].append(oracle_nmse)
            res_oracle_svd["psnr_stand_with_same_minmax"].append(oracle_psnr)
            res_oracle_svd["ssim_stand_with_same_minmax"].append(oracle_ssim)

            res_svd_fusi["nmse"].append(svd_limited_nmse)
            res_svd_fusi["psnr_stand_with_same_minmax"].append(svd_limited_psnr)
            res_svd_fusi["ssim_stand_with_same_minmax"].append(svd_limited_ssim)

    return res_method, res_oracle_svd, res_svd_fusi


def main():
    print("Predict on human reduced-sampling.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    num_frames_to_test = params.num_frames

    cudan = params.cudan
    device = torch.device(f"cuda:{cudan}" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}!")

    test_name = params.test_data_path.split("/")[-1]
    trianed_data_name = "L10_{}".format(params.model_path.split("/")[-1].split("_")[2])

    print("test_name", test_name)
    print("trianed_data_name", trianed_data_name)
    if test_name == "L10_124221":
        params.end_bin = 320
    else:
        params.end_bin = 310

    if trianed_data_name == test_name:
        params.start_bin = int(params.model_path.split("data")[0].split("_")[-1]) + 1

    print("params.start_bin", params.start_bin)

    # load model params ------------------------------------------------------#
    model_params = pickle.load(
        open(os.path.join(params.model_path, "params.pickle"), "rb")
    )
    params.model = model_params.model

    params.num_frames = num_frames_to_test

    print("number of frames to test is {}".format(params.num_frames))

    params.interleaved = int(np.floor(params.num_frames_full / params.num_frames))

    out_path_general = os.path.join(params.model_path, "predicted_metrics")
    if not os.path.exists(out_path_general):
        os.makedirs(out_path_general)

    print("out_path_general", out_path_general)

    # compute standardization constant --------------------------------------#
    (
        standardization_constant_folder,
        standardization_constant_name,
    ) = model_params.standardization_constant_path.split("/")[-2:]

    model_params.standardization_constant_path = os.path.join(
        "../data/standardization_constants",
        standardization_constant_folder,
        standardization_constant_name,
    )
    standardization_constant = utils.get_standardization_constant(model_params)

    if params.vis_svd:
        dataset_for_svd = getattr(datasetloader, params.loader.svd_function)(
            params.test_data_path,
            svd_with_respect_to_full_frames=params.svd_with_respect_to_full_frames_limited,
            num_frames=params.num_frames,
            interleaved=params.interleaved,
            num_frames_full=params.num_frames_full,
            num_channels=params.num_channels,
            remove_top_pixels=params.remove_top_pixels,
        )
    else:
        dataset_for_svd = None

    if params.vis_svd_full:
        dataset_for_svd_full_frames = getattr(
            datasetloader, params.loader.svd_function
        )(
            params.test_data_path,
            svd_with_respect_to_full_frames=params.svd_with_respect_to_full_frames,
            num_frames=params.num_frames_full,
            interleaved=1,
            num_frames_full=params.num_frames_full,
            num_channels=params.num_channels,
            remove_top_pixels=params.remove_top_pixels,
        )
    else:
        dataset_for_svd_full_frames = None

    # create dataset and dataloader ------------------------------------------#
    if params.model == "VARSfUSI":
        transform = [
            utils.FUSCenterCrop(params.crop_dim),
        ]
        dataset_for_method = getattr(
            datasetloader, params.loader.no_function_for_guidedsvd
        )(
            params.test_data_path,
            svd=params.svd_with_respect_to_full_frames_limited,
            num_frames=params.num_frames,
            iq_signal_mode=model_params.iq_signal_mode,
            interleaved=params.interleaved,
            standardization_constant=standardization_constant,
            transform=transform,
            num_frames_full=params.num_frames_full,
            num_channels=params.num_channels,
            total_frames=params.num_frames_full,
            remove_top_pixels=params.remove_top_pixels,
        )
    else:
        transform = [
            torchvision.transforms.CenterCrop(params.crop_dim),
        ]
        dataset_for_method = getattr(datasetloader, params.loader.no_function)(
            params.test_data_path,
            num_frames=params.num_frames,
            iq_signal_mode=model_params.iq_signal_mode,
            interleaved=params.interleaved,
            standardization_constant=standardization_constant,
            transform=transform,
            num_frames_full=params.num_frames_full,
            num_channels=params.num_channels,
            remove_top_pixels=params.remove_top_pixels,
        )

    # create model ---------------------------------------------------------#
    net, model_params = utils.get_model(model_params)

    print(params.model_type)
    checkpoint_path = os.path.join(
        params.model_path, "model", f"model_{params.model_type}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)

    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    net.to(device)

    pred_path = os.path.join(
        out_path_general,
        "pred",
        f"{test_name}",
    )
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    params.pred_path = pred_path

    # predict  ---------------------------------------------------------------#
    test_path_no = os.path.join(
        out_path_general,
        f"{test_name}_{params.num_frames}frames_reducedsampling_{params.model_type}_method.pt",
    )
    test_path_oracle_svd = os.path.join(
        out_path_general,
        f"{test_name}_{params.num_frames}frames_reducedsampling_{params.model_type}_oraclesvd.pt",
    )
    test_path_svd_fusi = os.path.join(
        out_path_general,
        f"{test_name}_{params.num_frames}frames_reducedsampling_{params.model_type}_svdfusi.pt",
    )

    if not os.path.exists(test_path_no):
        res_method, res_oracle_svd, res_svd_fusi = compute_loss(
            dataset_for_svd,
            dataset_for_svd_full_frames,
            dataset_for_method,
            net,
            standardization_constant,
            params,
        )

        torch.save(res_method, test_path_no)
        torch.save(res_oracle_svd, test_path_oracle_svd)
        torch.save(res_svd_fusi, test_path_svd_fusi)
    else:
        res_method = torch.load(test_path_no)
        res_oracle_svd = torch.load(test_path_oracle_svd)
        res_svd_fusi = torch.load(test_path_svd_fusi)

    print(
        "res_method nmse",
        np.round(np.mean(res_method["nmse"]), 4),
        "psnr",
        np.round(np.mean(res_method["psnr_stand_with_same_minmax"]), 4),
        "ssim",
        np.round(np.mean(res_method["ssim_stand_with_same_minmax"]), 4),
    )
    print(
        "res_oracle_svd nmse",
        np.round(np.mean(res_oracle_svd["nmse"]), 4),
        "psnr",
        np.round(np.mean(res_oracle_svd["psnr_stand_with_same_minmax"]), 4),
        "ssim",
        np.round(np.mean(res_oracle_svd["ssim_stand_with_same_minmax"]), 4),
    )
    print(
        "res_svd_fusi nmse",
        np.round(np.mean(res_svd_fusi["nmse"]), 4),
        "psnr",
        np.round(np.mean(res_svd_fusi["psnr_stand_with_same_minmax"]), 4),
        "ssim",
        np.round(np.mean(res_svd_fusi["ssim_stand_with_same_minmax"]), 4),
    )

    print(f"{params.model_path}")
    print(f"test on {test_name}")
    print("#########\n")
    print("done.")


if __name__ == "__main__":
    main()
