"""
Copyright (c) 2025 Bahareh Tolooshams

create train test testout multimice

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
from tqdm import tqdm
import argparse
import os

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out-path",
        type=str,
        help="out path",
        default="/data/fus/multimice",
    )
    parser.add_argument(
        "--animal-out",
        type=str,
        help="animal out for testing",
        default="Mouse_2ABR",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="/data/fus/multimice/all",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="number of frames",
        default=300,
    )
    parser.add_argument(
        "--train-test-split",
        type=list,
        help="list for train test split ratio",
        default=[0.9, 0.10],
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print("Delay.")

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()
    out_path = params["out_path"]
    data_path = params["data_path"]

    # create folder for out_path ---------------------------------------------#
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if not os.path.exists(f"{out_path}/train"):
        os.makedirs(f"{out_path}/train")

    if not os.path.exists(f"{out_path}/test"):
        os.makedirs(f"{out_path}/test")

    if not os.path.exists(f"{out_path}/test_out"):
        os.makedirs(f"{out_path}/test_out")

    # create dataset and dataloader ------------------------------------------#
    dataset = datasetloader.FUSdataset(
        params["data_path"], num_frames=params["num_frames"]
    )

    # -------------------------------------------------------------------------#
    animal_out = params["animal_out"]
    print(f"animal_out is {animal_out}")

    data_dict = dict()
    for data_path_curr in dataset.data_path_list:
        filename = data_path_curr.split("/")[-1]
        print(data_path_curr)
        mouse_name, functional_name, bin_name = filename.split("-")

        bin_num = bin_name.split("_")[2]
        delay_num = bin_name.split("_")[-1][:-3]

        if mouse_name not in data_dict:
            data_dict[f"{mouse_name}"] = dict()

        if functional_name not in data_dict[f"{mouse_name}"]:
            data_dict[f"{mouse_name}"][f"{functional_name}"] = dict()

        if delay_num not in data_dict[f"{mouse_name}"][f"{functional_name}"]:
            data_dict[f"{mouse_name}"][f"{functional_name}"][f"{delay_num}"] = list()

        data_dict[f"{mouse_name}"][f"{functional_name}"][f"{delay_num}"].append(
            f"{bin_num}"
        )
    print(data_dict)

    for mouse_name in data_dict.keys():
        # this is for testing out
        if mouse_name == animal_out:
            for functional_name in data_dict[f"{mouse_name}"].keys():
                print("test out", mouse_name, functional_name)
                for delay_num in data_dict[f"{mouse_name}"][
                    f"{functional_name}"
                ].keys():
                    bin_num_list = data_dict[f"{mouse_name}"][f"{functional_name}"][
                        f"{delay_num}"
                    ]
                    for bin_num in bin_num_list:
                        filename = f"{mouse_name}-{functional_name}-fUS_block_{bin_num}_{delay_num}.pt"
                        data_cur = torch.load(f"{data_path}/{filename}")

                        torch.save(data_cur, f"{out_path}/test_out/{filename}")

        # this is for train/test
        else:
            for functional_name in data_dict[f"{mouse_name}"].keys():
                print("train/test", mouse_name, functional_name)
                for delay_num in data_dict[f"{mouse_name}"][
                    f"{functional_name}"
                ].keys():
                    bin_num_list = data_dict[f"{mouse_name}"][f"{functional_name}"][
                        f"{delay_num}"
                    ]

                    num_bins = len(bin_num_list)
                    num_train = int(np.floor(num_bins * params["train_test_split"][0]))

                    for bin_num in bin_num_list:
                        filename = f"{mouse_name}-{functional_name}-fUS_block_{bin_num}_{delay_num}.pt"
                        data_cur = torch.load(f"{data_path}/{filename}")

                        if int(bin_num) < num_train:
                            torch.save(data_cur, f"{out_path}/train/{filename}")
                        else:
                            torch.save(data_cur, f"{out_path}/test/{filename}")

    print(f"data is saved at {out_path} in the folders")


if __name__ == "__main__":
    main()
