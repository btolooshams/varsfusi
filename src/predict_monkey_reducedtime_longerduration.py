"""
Copyright (c) 2025 Bahareh Tolooshams

predict monkey reduced-time longer duration

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
import pickle
import pytorch_msssim

import datasetloader, utils, metrics


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res-dir",
        type=str,
        help="results dir",
        default="../results",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="model name which resutl folders starts with",
        default="xxx",
    )
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

    parser.add_argument("--cudan", type=int, help="cuda number", default=0)

    params = parser.parse_args()

    return params


def compute_loss(
    net,
    params,
    dataloader,
    standardization_constant_already_applied,
    standardization_constant_for_eval,
    device="cpu",
    disable=True,
):
    net.eval()

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

    if dataloader:
        with torch.no_grad():
            batch_ctr = 0

            for idx, loaded_data in tqdm(enumerate(dataloader), disable=disable):

                if params.model == "VARSfUSI":
                    (iq_signal, dop_signal, dop_signal_with_svd, _) = loaded_data
                    dop_signal_with_svd = dop_signal_with_svd.to(device)

                else:
                    (iq_signal, dop_signal, _) = loaded_data

                if params.model == "DeepfUS":
                    _, _, w, _ = iq_signal.shape
                    iq_signal = torch.nn.functional.interpolate(
                        iq_signal, size=(w, params.num_frames), mode="bicubic"
                    )

                batch_ctr += 1

                iq_signal = iq_signal.to(device)
                dop_signal = dop_signal.to(device)

                dop_signal = torch.unsqueeze(dop_signal, dim=1)

                dur_increased_factor = int(
                    params.num_frames_to_test / params.num_frames
                )
                if dur_increased_factor == 1:
                    dur_increased_factor = None

                # forward encoder
                if params.model == "VARSfUSI":
                    dop_signal_est = net(
                        iq_signal,
                        dop_signal_with_svd,
                        dur_increased_factor=dur_increased_factor,
                    )
                else:
                    if "deepfus" in params.model_name:
                        dop_signal_est = net(iq_signal)
                    else:
                        dop_signal_est = net(
                            iq_signal, dur_increased_factor=dur_increased_factor
                        )

                ##### this is to make sure they are on the same range as results from training
                dop_signal = utils.unstandardize_general(
                    dop_signal,
                    standardization_constant_already_applied["dop_mean"],
                    standardization_constant_already_applied["dop_std"],
                )
                dop_signal_est = utils.unstandardize_general(
                    dop_signal_est,
                    standardization_constant_already_applied["dop_mean"],
                    standardization_constant_already_applied["dop_std"],
                )

                dop_signal = utils.standardize_general(
                    dop_signal,
                    standardization_constant_for_eval["dop_mean"],
                    standardization_constant_for_eval["dop_std"],
                )
                dop_signal_est = utils.standardize_general(
                    dop_signal_est,
                    standardization_constant_for_eval["dop_mean"],
                    standardization_constant_for_eval["dop_std"],
                )

                res["nmse"].append(metrics.nmse(dop_signal, dop_signal_est).item())
                res["psnr_stand_with_same_minmax"].append(
                    metrics.psnr_stand_with_same_minmax(
                        dop_signal, dop_signal_est
                    ).item()
                )
                res["ssim_stand_with_same_minmax"].append(
                    metrics.ssim_stand_with_same_minmax(
                        dop_signal, dop_signal_est, ssim_criterion
                    ).item()
                )

                print(np.mean(res["psnr_stand_with_same_minmax"]), np.mean(res["nmse"]))

    return res


def main():
    print("Predict on FUSi dataset.")
    print("Lower time frame.")

    num_frames_to_test = 32
    pad = "lowres_interp"
    checkpoint_type = "final"

    # init parameters -------------------------------------------------------#
    params_init = init_params()

    params_init.crop_dim = [136, 136]
    svd_for_limited = 75
    if num_frames_to_test == 64:
        svd_for_limited = 70

    for test_data_path in params_init.test_data_path_list:
        params_init.test_data_path = test_data_path

        params.test_name = "{}{}".format(
            params.test_data_path.split("/")[-2], params.test_data_path.split("/")[-1]
        )

        params_init.num_frames_to_test = num_frames_to_test
        params_init.checkpoint_type = checkpoint_type
        params_init.pad = pad

        disable = False

        res_path_list = os.listdir(params_init.res_dir)
        res_path_list = [
            f"{params_init.res_dir}/{x}"
            for x in res_path_list
            if f"{params_init.model_name}" in x
        ]

        for res_path in res_path_list:

            print(f"{checkpoint_type}, {pad}: {res_path}")
            print(f"test on {params_init.test_name}")

            # take parameters from the result path
            params = pickle.load(open(os.path.join(res_path, "params.pickle"), "rb"))

            out_path_general = os.path.join(res_path, "predicted_metrics")
            if not os.path.exists(out_path_general):
                os.makedirs(out_path_general)

            cudan = params_init.cudan
            device = torch.device(
                f"cuda:{cudan}" if torch.cuda.is_available() else "cpu"
            )

            for key in params_init.__dict__:
                params[key] = params_init.__dict__[key]

            if pad == "lowres_noproc":
                if params.num_frames / params.num_frames_to_test > 2.01:
                    continue

            print(
                f"number of frames: training {params.num_frames}, testing {params.num_frames_to_test}"
            )
            print(res_path)

            # compute standardization constant --------------------------------------#
            standardization_constant_already_applied = (
                utils.get_standardization_constant(params)
            )
            # this is being used to make sure that metrics such as (NMSE, ...) are similar across different methods
            # this is only used at the end for evaluation.
            standardization_constant_for_eval = torch.load(
                params.standardization_constant_path_only_for_evaluation
            )

            # create transforms for dataset -----------------------------------------#
            transform_inference = [
                utils.FUSCenterCrop(params_init.crop_dim),
            ]

            # create dataset and dataloader ------------------------------------------#
            if params.model == "VARSfUSI":
                stride_for_svd = 1
                test_dataset = datasetloader.FUSdatasetIQDOPsvdGeneral(
                    params_init.test_data_path,
                    svd=svd_for_limited,
                    num_frames=params.num_frames_to_test,
                    iq_signal_mode=params.iq_signal_mode,
                    standardization_constant=standardization_constant_already_applied,
                    transform=transform_inference,
                    stride_for_svd=stride_for_svd,
                )
            else:
                test_dataset = datasetloader.FUSdataset(
                    params_init.test_data_path,
                    num_frames=params.num_frames_to_test,
                    iq_signal_mode=params.iq_signal_mode,
                    standardization_constant=standardization_constant_already_applied,
                    transform=transform_inference,
                )

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=1,
                num_workers=params.loader.num_workers,
            )

            # create model ---------------------------------------------------------#
            net, params = utils.get_model(params)

            checkpoint_path = os.path.join(
                res_path, "model", f"model_{params_init.checkpoint_type}.pt"
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)

            net.load_state_dict(checkpoint["model_state_dict"])
            net.eval()
            net.to(device)

            # predict  ---------------------------------------------------------------#
            test_path = os.path.join(
                out_path_general,
                f"{params_init.test_name}_{params.num_frames_to_test}frames_reducedtime_{params.pad}_{params_init.checkpoint_type}_samesamplingres.pt",
            )

            if not os.path.exists(test_path):
                test_res_dict = compute_loss(
                    net,
                    params,
                    test_loader,
                    standardization_constant_already_applied,
                    standardization_constant_for_eval,
                    device,
                    disable,
                )
                torch.save(test_res_dict, test_path)
            else:
                test_res_dict = torch.load(test_path)

            print(f"{checkpoint_type}, {pad}: {res_path}")
            print(f"test on {params_init.test_name}")
            print("psnr", np.mean(test_res_dict["psnr_stand_with_same_minmax"]))
            print("ssim", np.mean(test_res_dict["ssim_stand_with_same_minmax"]))
            print("nmse", np.mean(test_res_dict["nmse"]))
            print("#########\n")

    print("done.")


if __name__ == "__main__":
    main()
