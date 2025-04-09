"""
Copyright (c) 2025 Bahareh Tolooshams

create data in pt format from iq bin dop mat files

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-path",
        type=str,
        help="output path to save data",
        default="/data/fus/multimice/all",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="/data/fus/multimice",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="number of frames",
        default=300,
    )
    parser.add_argument(
        "--num-frames-for-dop",
        type=int,
        help="number of frames",
        default=300,
    )
    parser.add_argument(
        "--delayshiftmax",
        type=int,
        help="delayshiftmax",
        default=0,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size",
        default=1,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="number of workers for dataloader",
        default=4,
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print("Create pt FUSI dataset from bin iq and mat dop.")

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()
    out_path = params["out_path"]

    # create folder for out_path ---------------------------------------------#
    if not os.path.exists(params["out_path"]):
        os.makedirs(params["out_path"])

    # create dataset and dataloader ------------------------------------------#
    dataset = datasetloader.FUSdatasetOnlineDOP(
        params["data_path"],
        num_frames=params["num_frames"],
        num_frames_for_dop=params["num_frames_for_dop"],
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    # visualize data and save ------------------------------------------------#
    for delayshift in range(0, params["delayshiftmax"] + 1, 5):
        print(f"delayshift is {delayshift}")

        data_loader.dataset.delayshift = delayshift

        for idx, (iq_signal, dop_signal, curr_input_path) in tqdm(
            enumerate(data_loader), disable=False
        ):
            for ex in range(iq_signal.shape[0]):
                filename = (curr_input_path[ex].split("Mouse_")[-1]).split(".b")[0]
                filename = filename.replace("/", "-")
                outname = f"{out_path}/Mouse_{filename}_delayshift{delayshift}.pt"

                data = dict()
                data["iq_signal"] = iq_signal[ex]
                data["dop_signal"] = dop_signal[ex]

                torch.save(data, outname)

    print("done!")
    print(f"data is saved at {out_path}")


if __name__ == "__main__":
    main()
