"""
Copyright (c) 2025 Bahareh Tolooshams

create test train monkey

:author: Bahareh Tolooshams
"""

import torch
import argparse
import os

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="/data/fus/monkey/S1/all",
        # default="/data/fus/monkey/S2/all",
        # default="/data/fus/monkey/S3/all",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="number of frames",
        default=250,
    )
    parser.add_argument(
        "--num_train",
        type=int,
        help="num train",
        default=120,
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print("Delay.")

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()

    data_path = params["data_path"]
    out_path = data_path[:-4]

    # create folder for out_path ---------------------------------------------#
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if not os.path.exists(f"{out_path}/train"):
        os.makedirs(f"{out_path}/train")

    if not os.path.exists(f"{out_path}/test"):
        os.makedirs(f"{out_path}/test")

    # create dataset and dataloader ------------------------------------------#
    dataset = datasetloader.FUSdataset(
        params["data_path"], num_frames=params["num_frames"]
    )

    # -------------------------------------------------------------------------#
    data_dict = dict()
    for data_path_curr in dataset.data_path_list:
        # filename has this form fUS_block_{bin_num}_{delay_num}.pt
        filename = data_path_curr.split("/")[-1]
        _, _, bin_num, delay_name = filename.split("_")

        delay_num = delay_name.split(".")[0]

        if delay_num not in data_dict:
            data_dict[f"{delay_num}"] = list()

        data_dict[f"{delay_num}"].append(f"{bin_num}")

    for delay_num in data_dict.keys():
        bin_num_list = data_dict[f"{delay_num}"]

        num_bins = len(bin_num_list)
        num_train = params["num_train"]
        num_test = num_bins - num_train

        print(
            "delay_num",
            delay_num,
            "num_bins",
            num_bins,
            "num_train",
            num_train,
            "num_test",
            num_test,
        )

        for bin_num in bin_num_list:
            filename = f"fUS_block_{bin_num}_{delay_num}.pt"
            data_cur = torch.load(f"{data_path}/{filename}")

            if int(bin_num) <= num_train:
                torch.save(data_cur, f"{out_path}/train/{filename}")
            else:
                torch.save(data_cur, f"{out_path}/test/{filename}")

    print(f"data is saved at {out_path} in train/test folders")


if __name__ == "__main__":
    main()
