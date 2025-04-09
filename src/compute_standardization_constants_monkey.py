"""
Copyright (c) 2025 Bahareh Tolooshams

compute std constant monkey

:author: Bahareh Tolooshams
"""

import torch
import argparse
import os

import datasetloader, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-path",
        type=str,
        help="output path to save normalization constatnt",
        default="/data/fus/monkey/S1",
    )
    parser.add_argument(
        "--iq-signal-mode",
        type=str,
        help="e.g., real, abs, complex, stack",
        default="stack",
    )
    parser.add_argument(
        "--take-random-window",
        type=bool,
        help="take random window of frames",
        default=True,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="/data/fus/monkey/S1/train",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="number of frames",
        default=32,
    )
    parser.add_argument(
        "--crop-dim",
        type=list,
        help="crop dimension of input image",
        default=[136, 136],
    )
    parser.add_argument(
        "--degrees",
        type=list,
        help="rotation degrees",
        default=[-10, 10],
    )
    parser.add_argument(
        "--degrees_flag",
        type=bool,
        help="rotation flag",
        default=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size",
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="number of workers for dataloader",
        default=16,
    )
    parser.add_argument(
        "--scalar-flag",
        type=bool,
        help="scalar flag",
        default=True,
    )
    parser.add_argument(
        "--pixelwise-flag",
        type=bool,
        help="pixelwise flag",
        default=False,
    )
    parser.add_argument(
        "--pixelwisetime-flag",
        type=bool,
        help="pixelwise time flag",
        default=False,
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print("Compute standardization constant for a FUSi dataset.")

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()
    out_path = params["out_path"]

    iq_signal_mode = params["iq_signal_mode"]
    folder_name = params["data_path"].split("/")[-1]
    num_frames = params["num_frames"]
    crop_dim0 = params["crop_dim"][0]
    out_name = f"{folder_name}_{iq_signal_mode}_{num_frames}_{crop_dim0}"

    # create folder for out_path ---------------------------------------------#
    if not os.path.exists(params["out_path"]):
        os.makedirs(params["out_path"])

    # create dataset and dataloader ------------------------------------------#
    # this is to make sure all data has this dim
    if params["degrees_flag"]:
        transform = [
            utils.FUSRandomCrop(params["crop_dim"], pad_if_needed=True),
            utils.FUSRandomRotation(degrees=params["degrees"]),
            utils.FUSRandomHorizontalFlip(p=0.5),
            utils.FUSRandomVerticalFlip(p=0.5),
        ]
    else:
        transform = [
            utils.FUSRandomCrop(params["crop_dim"], pad_if_needed=True),
            utils.FUSRandomHorizontalFlip(p=0.5),
            utils.FUSRandomVerticalFlip(p=0.5),
        ]

    dataset = datasetloader.FUSdataset(
        params["data_path"],
        num_frames=params["num_frames"],
        iq_signal_mode=params["iq_signal_mode"],
        take_random_window=params["take_random_window"],
        transform=transform,
    )

    # -------------------------------------------------------------------------#
    if params["scalar_flag"]:
        scalar_standardization_constant = utils.compute_scalar_standardization_constant(
            dataset,
            iq_signal_mode=params["iq_signal_mode"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
        )

        print(f"standardization constant have dimesinos ...")

        print("iq_mean:", scalar_standardization_constant["iq_mean"].shape)
        print("iq_std:", scalar_standardization_constant["iq_std"].shape)
        print("dop_mean:", scalar_standardization_constant["dop_mean"].shape)
        print("dop_std:", scalar_standardization_constant["dop_std"].shape)

        out_file = f"{out_path}/{out_name}_withtransformation_scalar_standardization_constant.pt"
        torch.save(scalar_standardization_constant, out_file)
        print(f"scalar standardization constant are saved at {out_file}!")

    # -------------------------------------------------------------------------#
    if params["pixelwise_flag"]:
        pixelwise_standardization_constant = (
            utils.compute_pixelwise_standardization_constant(
                dataset,
                iq_signal_mode=params["iq_signal_mode"],
                batch_size=params["batch_size"],
                num_workers=params["num_workers"],
            )
        )

        print(f"standardization constant have dimesinos ...")

        print("iq_mean:", pixelwise_standardization_constant["iq_mean"].shape)
        print("iq_std:", pixelwise_standardization_constant["iq_std"].shape)
        print("dop_mean:", pixelwise_standardization_constant["dop_mean"].shape)
        print("dop_std:", pixelwise_standardization_constant["dop_std"].shape)

        out_file = f"{out_path}/{out_name}_withtransformation_pixelwise_standardization_constant.pt"
        torch.save(pixelwise_standardization_constant, out_file)
        print(f"pixelwise standardization constant are saved at {out_file}!")

    # -------------------------------------------------------------------------#
    if params["pixelwisetime_flag"]:
        pixelwise_time_standardization_constant = (
            utils.compute_pixelwiseplustime_standardization_constant(
                dataset,
                iq_signal_mode=params["iq_signal_mode"],
                batch_size=params["batch_size"],
                num_workers=params["num_workers"],
            )
        )

        print(f"standardization constant have dimesinos ...")

        print("iq_mean:", pixelwise_time_standardization_constant["iq_mean"].shape)
        print("iq_std:", pixelwise_time_standardization_constant["iq_std"].shape)
        print("dop_mean:", pixelwise_time_standardization_constant["dop_mean"].shape)
        print("dop_std:", pixelwise_time_standardization_constant["dop_std"].shape)

        out_file = f"{out_path}/{out_name}_withtransformation_pixelwiseplustime_standardization_constant.pt"
        torch.save(pixelwise_time_standardization_constant, out_file)
        print(f"pixelwise plus time standardization constant are saved at {out_file}!")


if __name__ == "__main__":
    main()
