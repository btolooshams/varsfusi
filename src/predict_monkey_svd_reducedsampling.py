"""
Copyright (c) 2025 Bahareh Tolooshams

prediction script

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
import os
import pytorch_msssim

import datasetloader, utils, metrics


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-data-path-list",
        type=list,
        help="test data path list",
        default=[
            # this is for monkey new data used for decoding
            "/data/fus/monkey/S1/test",
            # "/data/fus/monkey/S2/test",
            # "/data/fus/monkey/S3/test",
        ],
    )
    parser.add_argument(
        "--standardization_constant_path_only_for_evaluation",
        type=str,
        help="standardization_constant_path_only_for_evaluation",
        default="/data/fus/multimice/train_real_32_128_withtransformation_scalar_standardization_constant.pt",
    )
    params = parser.parse_args()

    return params


def compute_loss_svd(
    params, dataloader, standardization_constant, device="cpu", disable=True
):
    ssim_criterion = pytorch_msssim.SSIM(
        data_range=1,
        size_average=True,
        channel=1,
        win_size=7,
    )

    res = {
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

    if dataloader:
        with torch.no_grad():
            batch_ctr = 0

            for idx, (
                iq_signal,
                dop_signal,
                dop_signal_est_with_limited_frame,
                dop_signal_lower_limit,
                curr_input_path,
            ) in tqdm(enumerate(dataloader), disable=disable):
                batch_ctr += 1

                iq_signal = iq_signal.to(device)
                dop_signal = dop_signal.to(device)
                dop_signal_est_with_limited_frame = (
                    dop_signal_est_with_limited_frame.to(device)
                )
                dop_signal_lower_limit = dop_signal_lower_limit.to(device)

                dop_signal = torch.unsqueeze(dop_signal, dim=1)
                dop_signal_est_with_limited_frame = torch.unsqueeze(
                    dop_signal_est_with_limited_frame, dim=1
                )
                dop_signal_lower_limit = torch.unsqueeze(dop_signal_lower_limit, dim=1)

                ##### this is to make sure they are on the same range as results from training
                dop_signal = utils.standardize_general(
                    dop_signal,
                    standardization_constant["dop_mean"],
                    standardization_constant["dop_std"],
                )
                dop_signal_est_with_limited_frame = utils.standardize_general(
                    dop_signal_est_with_limited_frame,
                    standardization_constant["dop_mean"],
                    standardization_constant["dop_std"],
                )
                dop_signal_lower_limit = utils.standardize_general(
                    dop_signal_lower_limit,
                    standardization_constant["dop_mean"],
                    standardization_constant["dop_std"],
                )

                res["nmae"].append(
                    metrics.nmae(dop_signal, dop_signal_est_with_limited_frame).item()
                )
                res["nmse"].append(
                    metrics.nmse(dop_signal, dop_signal_est_with_limited_frame).item()
                )
                res["psnr_stand_with_same_minmax"].append(
                    metrics.psnr_stand_with_same_minmax(
                        dop_signal, dop_signal_est_with_limited_frame
                    ).item()
                )
                res["ssim_stand_with_same_minmax"].append(
                    metrics.ssim_stand_with_same_minmax(
                        dop_signal, dop_signal_est_with_limited_frame, ssim_criterion
                    ).item()
                )

                ##########
                ### dop_signal_lower_limit

                res_oracle_svd["nmae"].append(
                    metrics.nmae(dop_signal, dop_signal_lower_limit).item()
                )
                res_oracle_svd["nmse"].append(
                    metrics.nmse(dop_signal, dop_signal_lower_limit).item()
                )
                res_oracle_svd["psnr_stand_with_same_minmax"].append(
                    metrics.psnr_stand_with_same_minmax(
                        dop_signal, dop_signal_lower_limit
                    ).item()
                )
                res_oracle_svd["ssim_stand_with_same_minmax"].append(
                    metrics.ssim_stand_with_same_minmax(
                        dop_signal, dop_signal_lower_limit, ssim_criterion
                    ).item()
                )

    return res, res_oracle_svd


def main():
    print("Predict on fUS dataset with SVD method.")

    num_frames_list = [32]

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()

    params.iq_signal_mode = "complex"
    params.decimation_factor = 1
    params.cudan = 0

    params.crop_dim = [136, 136]
    datasetname = "FUSdatasetforSVDGeneralinterleaved"
    total_number_of_frames = 250
    svd = 38
    if params.decimation_factor == 2:
        pass
    elif params.decimation_factor == 1:
        svd_for_limited = 125

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    for test_data_path in params.test_data_path_list:
        params.test_data_path = test_data_path

        params.test_name = "{}{}".format(
            params.test_data_path.split("/")[-2], params.test_data_path.split("/")[-1]
        )

        standardization_constant_only_for_evaluation = torch.load(
            params.standardization_constant_path_only_for_evaluation
        )

        cudan = params.cudan
        device = torch.device(f"cuda:{cudan}" if torch.cuda.is_available() else "cpu")
        print(f"running on {device}!")
        disable = False

        out_path = os.path.join(
            "..", "results", "svd_{}_{}".format(params.test_name, random_date)
        )
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # compute standardization constant --------------------------------------#
        standardization_constant = None

        transform = [
            utils.FUSCenterCrop(params.crop_dim),
        ]

        for num_frames in num_frames_list:

            params.num_frames = num_frames
            params.interleaved = int(
                np.floor(total_number_of_frames / params.num_frames)
            )

            # create dataset and dataloader ------------------------------------------#
            test_dataset = getattr(datasetloader, datasetname)(
                params.test_data_path,
                svd=svd,
                svd_for_limited=svd_for_limited,
                num_frames=params.num_frames,
                iq_signal_mode=params.iq_signal_mode,
                interleaved=params.interleaved,
                standardization_constant=standardization_constant,
                take_random_window=False,
                transform=transform,
            )

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=1,
                num_workers=16,
            )

            # predict  ---------------------------------------------------------------#
            print(
                f"{params.test_name}, {params.num_frames} frames with {params.decimation_factor} decimation factor."
            )

            r_path = os.path.join(
                out_path,
                f"{params.test_name}_{params.num_frames}frames_reducedsampling_{params.decimation_factor}decimation_svdfusi.pt",
            )

            r_path_ll = os.path.join(
                out_path,
                f"{params.test_name}_{params.num_frames}frames_reducedsampling_{params.decimation_factor}decimation_oraclesvd.pt",
            )

            if not os.path.exists(r_path) or not os.path.exists(r_path_ll):
                r_res_dict, r_res_dict_lower_limit = compute_loss_svd(
                    params,
                    test_loader,
                    standardization_constant_only_for_evaluation,
                    device,
                    disable,
                )

                torch.save(r_res_dict, r_path)
                torch.save(r_res_dict_lower_limit, r_path_ll)

                print(np.mean(r_res_dict["ssim_stand_with_same_minmax"]))
                print(np.mean(r_res_dict["psnr_stand_with_same_minmax"]))
                print(np.mean(r_res_dict["nmse"]))

            else:

                r_res_dict = torch.load(r_path)
                r_res_dict_lower_limit = torch.load(r_path_ll)

                print(np.mean(r_res_dict["ssim_stand_with_same_minmax"]))
                print(np.mean(r_res_dict["psnr_stand_with_same_minmax"]))
                print(np.mean(r_res_dict["nmse"]))

            print("done.")


if __name__ == "__main__":
    main()
