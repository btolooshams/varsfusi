"""
Copyright (c) 2025 Bahareh Tolooshams

create data in pt format from iq bin dop mat files monkey

:author: Bahareh Tolooshams
"""

import torch
import argparse
import os
from tqdm import tqdm

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="/data/fus/monkey/S1",
        # default="/data/fus/monkey/S2",
        # default="/data/fus/monkey/S3",
    )
    parser.add_argument(
        "--svd",
        type=float,
        help="threshold on svd",
        default=38,  # 15 % given the 250 frames
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="number of frames",
        default=250,
    )
    parser.add_argument(
        "--num-frames-for-dop",
        type=int,
        help="number of frames",
        default=250,
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
        default=24,
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print("Create pt fUS dataset for andersen-thierri from bin iq and mat dop.")

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()

    out_path = "{}/all".format(params["data_path"])

    # create folder for out_path ---------------------------------------------#
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # create dataset and dataloader ------------------------------------------#
    dataset = datasetloader.FUSdatasetOnlineDOPGeneral(
        params["data_path"],
        svd=params["svd"],
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
                filename = curr_input_path[ex].split("/")[-1].split(".")[0]
                outname = f"{out_path}/{filename}_delayshift{delayshift}.pt"

                data = dict()
                data["iq_signal"] = iq_signal[ex]
                data["dop_signal"] = dop_signal[ex]

                torch.save(data, outname)

    print("done!")
    print(f"data is saved at {out_path}")


if __name__ == "__main__":
    main()
