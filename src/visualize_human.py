"""
Copyright (c) 2025 Bahareh Tolooshams

visualize human

:author: Bahareh Tolooshams
"""

import numpy as np
import torch
import torchvision
import configmypy
import os
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

import datasetloader, utils

cudan = 0
device = torch.device(f"cuda:{cudan}" if torch.cuda.is_available() else "cpu")


def init_params():
    pipe = configmypy.ConfigPipeline(
        [
            configmypy.YamlConfig(
                config_file="./human_finetune_prediction_reducedtime.yaml",
                config_name="default",
                config_folder="../config",
            ),
        ]
    )
    params = pipe.read_conf()

    return params


def visualize(
    dataset_for_svd,
    dataset_for_svd_full_frames,
    dataset_for_no,
    net,
    standardization_constant,
    params,
):

    for ctr in [50, 75, 100, 120, 150, 175, 200, 250, 300]:

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

        dataset_for_no.data_path_list = [fusblock_path]
        dataset_for_no.num_data = 1

        # get from dataset_for_svd
        if dataset_for_svd is not None:
            _, dop_signal_svd, curr_input_path_svd = dataset_for_svd[0]
            dop_signal_svd = utils.standardize_general(
                dop_signal_svd,
                standardization_constant["dop_mean"],
                standardization_constant["dop_std"],
            )
            dop_signal_svd = dop_signal_svd.numpy()

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
            dop_signal_svd_full_frames = dop_signal_svd_full_frames.numpy()

        # get lower limit svd
        svd_org = params.svd_with_respect_to_full_frames
        iq_signal_for_ll = dataset_for_svd_full_frames[0][0]
        z_dim, x_dim, t_dim = iq_signal_for_ll.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_for_ll.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )
        iqf_full = U[:, svd_org:] @ np.diag(Λ[svd_org:]) @ V.T[svd_org:]
        iqf_lower_limit = iqf_full[:, : params.num_frames]
        dop_signal_svd_lowerlimit = np.mean(
            np.abs(iqf_lower_limit) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_svd_lowerlimit = torch.tensor(dop_signal_svd_lowerlimit)
        dop_signal_svd_lowerlimit = utils.standardize_general(
            dop_signal_svd_lowerlimit,
            standardization_constant["dop_mean"],
            standardization_constant["dop_std"],
        )
        dop_signal_svd_lowerlimit = dop_signal_svd_lowerlimit.numpy()

        # get from dataset_for_no
        if params.model == "VARSfUSI":
            (
                iq_signal,
                dop_signal_limited_frames,
                curr_input_path_no,
            ) = dataset_for_no[0]
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
                curr_input_path_no,
            ) = dataset_for_no[0]
            iq_signal = iq_signal.to(device)
            iq_signal = torch.unsqueeze(iq_signal, dim=0)
            dop_signal_no = net(iq_signal)
        dop_signal_no = torch.squeeze(dop_signal_no)
        dop_signal_no = dop_signal_no.detach().cpu().numpy()

        ######
        dop_signal_no = dop_signal_no[5:-6]

        a = np.min(dop_signal_svd_full_frames)
        b = np.percentile(dop_signal_svd_full_frames, 95)

        cmap = "hot"
        plot_dop(
            dop_signal_svd_full_frames,
            a,
            b,
            os.path.join(params.fig_path, f"dop_{ctr}.svg"),
            cmap=cmap,
        )
        plot_dop(
            dop_signal_no,
            a,
            b,
            os.path.join(params.fig_path, f"dop_method_{ctr}.svg"),
            cmap=cmap,
        )
        plot_dop(
            dop_signal_svd,
            a,
            b,
            os.path.join(params.fig_path, f"dop_svd_limited_{ctr}.svg"),
            cmap=cmap,
        )
        plot_dop(
            dop_signal_svd_lowerlimit,
            a,
            b,
            os.path.join(params.fig_path, f"dop_oracle_svd_{ctr}.svg"),
            cmap=cmap,
        )

        plot_dop(
            dop_signal_svd_full_frames,
            a,
            b,
            os.path.join(params.fig_path, f"dop_{ctr}.png"),
            cmap=cmap,
        )
        plot_dop(
            dop_signal_no,
            a,
            b,
            os.path.join(params.fig_path, f"dop_method_{ctr}.png"),
            cmap=cmap,
        )
        plot_dop(
            dop_signal_svd,
            a,
            b,
            os.path.join(params.fig_path, f"dop_svd_limited_{ctr}.png"),
            cmap=cmap,
        )
        plot_dop(
            dop_signal_svd_lowerlimit,
            a,
            b,
            os.path.join(params.fig_path, f"dop_oracle_svd_{ctr}.png"),
            cmap=cmap,
        )

    return


def plot_dop(
    dop,
    a,
    b,
    name,
    cmap="hot",
):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": False,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.remove()

    if a:
        if b:
            plt.imshow(dop, cmap=cmap, vmin=a, vmax=b)
            plt.axis("off")
    else:
        plt.imshow(dop, cmap=cmap)
        plt.axis("off")

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(
        name,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()


def main():
    print("Online Imaging.")

    # init parameters -------------------------------------------------------#
    params = init_params()

    cudan = params.cudan
    device = torch.device(f"cuda:{cudan}" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}!")

    test_name = params.test_data_path.split("/")[-1]
    trianed_data_name = "L10_{}".format(params.model_path.split("/")[-1].split("_")[2])

    print("test_name", test_name)
    print("trianed_data_name", trianed_data_name)
    if trianed_data_name == test_name:
        params.start_bin = int(params.model_path.split("data")[0].split("_")[-1]) + 1

    print("params.start_bin", params.start_bin)

    # load model params ------------------------------------------------------#
    model_params = pickle.load(
        open(os.path.join(params.model_path, "params.pickle"), "rb")
    )
    params.model = model_params.model

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
        dataset_for_no = getattr(
            datasetloader, params.loader.no_function_for_guidedsvd
        )(
            params.test_data_path,
            svd=params.svd_with_respect_to_full_frames_limited,
            num_frames=params.num_frames,
            iq_signal_mode=model_params.iq_signal_mode,
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
        dataset_for_no = getattr(datasetloader, params.loader.no_function)(
            params.test_data_path,
            num_frames=params.num_frames,
            iq_signal_mode=model_params.iq_signal_mode,
            standardization_constant=standardization_constant,
            transform=transform,
            num_frames_full=params.num_frames_full,
            num_channels=params.num_channels,
            remove_top_pixels=params.remove_top_pixels,
        )

    # create model ---------------------------------------------------------#
    net, model_params = utils.get_model(model_params)

    checkpoint_path = os.path.join(
        params.model_path, "model", f"model_{params.model_type}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)

    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    net.to(device)

    # visualize  -----------------------------------------------------------#
    fig_path = os.path.join(
        out_path_general,
        "vis",
        f"{test_name}",
    )
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    params.fig_path = fig_path

    visualize(
        dataset_for_svd,
        dataset_for_svd_full_frames,
        dataset_for_no,
        net,
        standardization_constant,
        params,
    )

    print(f"{params.model_path}")
    print(f"test on {test_name}")
    print("#########\n")
    print("done.")


if __name__ == "__main__":
    main()
