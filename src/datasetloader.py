"""
Copyright (c) 2025 Tolooshams

create dataloaders

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os

import utils


class FUSdataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        num_frames=300,
        iq_signal_mode="complex",
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
    ):
        self.data_path = data_path
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames

        self.filename_list = os.listdir(self.data_path)
        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # the data is a tensor
        data = torch.load(curr_input_path)

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, data["iq_signal"].shape[-1] - self.num_frames
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        iq_signal = data["iq_signal"][
            :,
            :,
            self.offset_to_take_frames : self.offset_to_take_frames + self.num_frames,
        ]
        dop_signal = data["dop_signal"]

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal = transform(iq_signal, dop_signal)
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            iq_signal, dop_signal = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        #### in general
        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        #### for stack it would be
        # iq_signal: (H, W, 2T)

        return iq_signal, dop_signal, curr_input_path


class FUSdatasetinterleaved(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        num_frames=300,
        iq_signal_mode="complex",
        interleaved=1,
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
    ):
        self.data_path = data_path
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.interleaved = interleaved
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames

        self.filename_list = os.listdir(self.data_path)
        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # the data is a tensor
        data = torch.load(curr_input_path)

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(0, self.num_frames)
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        iq_signal = data["iq_signal"][:, :, self.offset_to_take_frames :]
        iq_signal = iq_signal[:, :, :: self.interleaved]
        offset = np.random.randint(0, iq_signal.shape[-1] - self.num_frames + 1)
        iq_signal = iq_signal[:, :, offset : offset + self.num_frames]

        dop_signal = data["dop_signal"]

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal = transform(iq_signal, dop_signal)
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            iq_signal, dop_signal = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        #### in general
        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        #### for stack it would be
        # iq_signal: (H, W, 2T)

        return iq_signal, dop_signal, curr_input_path


class FUSdatasetIQDOPsvd(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        num_frames=300,
        iq_signal_mode="complex",
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
        total_frames=300,
        stride_for_svd=1,
        svd_for_limited=None,
    ):
        self.data_path = data_path
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames
        self.total_frames = total_frames
        self.stride_for_svd = stride_for_svd
        self.svd_for_limited = svd_for_limited

        self.filename_list = os.listdir(self.data_path)
        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        curr_folder_path = curr_input_path.split(".")[0].split("/")[-1]
        Mouse_id = (curr_input_path.split("Mouse_")[-1]).split("-")[0]
        if self.svd_for_limited is None:
            if Mouse_id == "1AAN":
                svd = 40
            elif Mouse_id == "1ABN":
                svd = 50
            elif Mouse_id == "2AAR":
                svd = 60
            elif Mouse_id == "2ABR":
                svd = 30
            elif Mouse_id == "A":
                svd = 60
            elif Mouse_id == "B":
                svd = 60
            else:
                raise ValueError(f"Mouse folder is not known!")
        else:
            svd = self.svd_for_limited

        # adjusting the svd based on the number of frames
        svd = int(np.floor(svd * (self.num_frames / self.total_frames)))

        # the data is a tensor
        data = torch.load(curr_input_path)

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, data["iq_signal"].shape[-1] - self.num_frames
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        iq_signal = data["iq_signal"][
            :,
            :,
            self.offset_to_take_frames : self.offset_to_take_frames + self.num_frames,
        ]
        dop_signal = data["dop_signal"]

        ##################################################################
        ##################################################################
        ##### to get from limited frames
        # desired output
        iq_signal_for_svd = iq_signal[:, :, :: self.stride_for_svd]
        z_dim, x_dim, t_dim = iq_signal_for_svd.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_for_svd.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )
        ##################################################################

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal, dop_signal_with_limited_time_frames
                )
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            (
                iq_signal,
                dop_signal,
                dop_signal_with_limited_time_frames,
            ) = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
                dop_from_svd=dop_signal_with_limited_time_frames,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        #### in general
        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        #### for stack it would be
        # iq_signal: (H, W, 2T)

        return (
            iq_signal,
            dop_signal,
            dop_signal_with_limited_time_frames,
            curr_input_path,
        )


class FUSdatasetIQDOPsvdGeneral(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        svd,
        num_frames=250,
        iq_signal_mode="complex",
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
        total_frames=250,
        stride_for_svd=1,
    ):
        self.data_path = data_path
        self.svd = svd
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames
        self.total_frames = total_frames
        self.stride_for_svd = stride_for_svd

        self.filename_list = os.listdir(self.data_path)
        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        curr_folder_path = curr_input_path.split(".")[0].split("/")[-1]

        # adjusting the svd based on the number of frames
        svd = int(np.floor(self.svd * (self.num_frames / self.total_frames)))

        # the data is a tensor
        data = torch.load(curr_input_path)

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, data["iq_signal"].shape[-1] - self.num_frames
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        iq_signal = data["iq_signal"][
            :,
            :,
            self.offset_to_take_frames : self.offset_to_take_frames + self.num_frames,
        ]
        dop_signal = data["dop_signal"]

        ##################################################################
        ##################################################################
        ##### to get from limited frames
        # desired output
        iq_signal_for_svd = iq_signal[:, :, :: self.stride_for_svd]
        z_dim, x_dim, t_dim = iq_signal_for_svd.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_for_svd.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )
        ##################################################################

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal, dop_signal_with_limited_time_frames
                )
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            (
                iq_signal,
                dop_signal,
                dop_signal_with_limited_time_frames,
            ) = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
                dop_from_svd=dop_signal_with_limited_time_frames,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        #### in general
        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        #### for stack it would be
        # iq_signal: (H, W, 2T)

        return (
            iq_signal,
            dop_signal,
            dop_signal_with_limited_time_frames,
            curr_input_path,
        )


class FUSdatasetIQDOPsvdGeneralinterleaved(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        svd,
        num_frames=250,
        iq_signal_mode="complex",
        interleaved=1,
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
        total_frames=250,
        stride_for_svd=1,
    ):
        self.data_path = data_path
        self.svd = svd
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.interleaved = interleaved
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames
        self.total_frames = total_frames
        self.stride_for_svd = stride_for_svd

        self.filename_list = os.listdir(self.data_path)
        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        curr_folder_path = curr_input_path.split(".")[0].split("/")[-1]

        # adjusting the svd based on the number of frames
        svd = int(np.floor(self.svd * (self.num_frames / self.total_frames)))

        # the data is a tensor
        data = torch.load(curr_input_path)

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, (self.total_frames - self.interleaved * self.num_frames)
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        iq_signal = data["iq_signal"][:, :, self.offset_to_take_frames :]
        iq_signal = iq_signal[:, :, :: self.interleaved]
        offset = np.random.randint(0, iq_signal.shape[-1] - self.num_frames + 1)
        iq_signal = iq_signal[:, :, offset : offset + self.num_frames]
        dop_signal = data["dop_signal"]

        ##################################################################
        ##################################################################
        ##### to get from limited frames

        # desired output
        iq_signal_for_svd = iq_signal[:, :, :: self.stride_for_svd]
        z_dim, x_dim, t_dim = iq_signal_for_svd.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_for_svd.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )
        ##################################################################

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal, dop_signal_with_limited_time_frames
                )
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            (
                iq_signal,
                dop_signal,
                dop_signal_with_limited_time_frames,
            ) = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
                dop_from_svd=dop_signal_with_limited_time_frames,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        #### in general
        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        #### for stack it would be
        # iq_signal: (H, W, 2T)

        return (
            iq_signal,
            dop_signal,
            dop_signal_with_limited_time_frames,
            curr_input_path,
        )


class FUSdatasetforSVDGeneral(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        svd,
        svd_for_limited,
        ratio_svd=1,
        num_frames=250,
        decimation_factor=1,
        iq_signal_mode="complex",
        standardization_constant=None,
        take_random_window=False,
        offset_to_take_frames=0,
        transform=None,
    ):
        print("FUSdatasetforSVDGeneral only works with deterministic transform!")

        self.data_path = data_path
        self.svd = svd
        self.svd_for_limited = svd_for_limited
        self.ratio_svd = ratio_svd
        self.num_frames = num_frames
        self.decimation_factor = decimation_factor
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.offset_to_take_frames = offset_to_take_frames
        self.transform = transform
        self.total_frames = 250

        self.filename_list = os.listdir(self.data_path)
        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        svd_org = self.svd * 1

        # adjusting the svd based on the number of frames
        svd = int(
            np.floor(
                self.svd_for_limited
                * self.ratio_svd
                * (self.num_frames / self.total_frames)
                / self.decimation_factor
            )
        )
        # the data is a tensor
        data = torch.load(curr_input_path)

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, data["iq_signal"].shape[-1] - self.num_frames
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        dop_signal = data["dop_signal"]

        ##################################################################
        ##################################################################
        ##### to get from full frames, after remove the clutter, only take power from the limited first frames
        iq_signal_full = data["iq_signal"]

        # desired output
        z_dim, x_dim, t_dim = iq_signal_full.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_full.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf_full = U[:, svd_org:] @ np.diag(Λ[svd_org:]) @ V.T[svd_org:]
        iqf_lower_limit = iqf_full[
            :, self.offset_to_take_frames : self.offset_to_take_frames + self.num_frames
        ]
        iqf_lower_limit = iqf_lower_limit[:, :: self.decimation_factor]

        dop_signal_lower_limit = np.mean(np.abs(iqf_lower_limit) ** 2, axis=-1).reshape(
            z_dim, x_dim
        )
        dop_signal_lower_limit = torch.tensor(
            dop_signal_lower_limit, dtype=torch.float32
        )

        ##################################################################
        ##################################################################
        ##### to get from limited frames
        iq_signal = data["iq_signal"][
            :,
            :,
            self.offset_to_take_frames : self.offset_to_take_frames + self.num_frames,
        ]
        iq_signal = iq_signal[:, :, :: self.decimation_factor]

        # desired output
        z_dim, x_dim, t_dim = iq_signal.shape
        U, Λ, V = np.linalg.svd(
            iq_signal.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal = transform(iq_signal, dop_signal)

                _, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal_with_limited_time_frames
                )

                _, dop_signal_lower_limit = transform(iq_signal, dop_signal_lower_limit)

            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            iq_signal, dop_signal = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
            )
            _, dop_signal_with_limited_time_frames = utils.standardize(
                iq_signal,
                dop_signal_with_limited_time_frames,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            _, dop_signal_lower_limit = utils.standardize(
                iq_signal,
                dop_signal_lower_limit,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        return (
            iq_signal,
            dop_signal,
            dop_signal_with_limited_time_frames,
            dop_signal_lower_limit,
            curr_input_path,
        )


class FUSdatasetforSVDGeneralinterleaved(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        svd,
        svd_for_limited,
        ratio_svd=1,
        num_frames=250,
        iq_signal_mode="complex",
        interleaved=1,
        standardization_constant=None,
        take_random_window=False,
        offset_to_take_frames=0,
        transform=None,
    ):
        print(
            "FUSdatasetforSVDGeneralinterleaved only works with deterministic transform!"
        )

        self.data_path = data_path
        self.svd = svd
        self.svd_for_limited = svd_for_limited
        self.ratio_svd = ratio_svd
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.interleaved = interleaved
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.offset_to_take_frames = offset_to_take_frames
        self.transform = transform
        self.total_frames = 250

        self.filename_list = os.listdir(self.data_path)
        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        svd_org = self.svd * 1

        # adjusting the svd based on the number of frames
        svd = int(
            np.floor(
                self.svd_for_limited
                * self.ratio_svd
                * (self.num_frames / self.total_frames)
            )
        )
        # the data is a tensor
        data = torch.load(curr_input_path)

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, (self.total_frames - self.interleaved * self.num_frames)
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        dop_signal = data["dop_signal"]

        ##################################################################
        ##################################################################
        ##### to get from full frames, after remove the clutter, only take power from the limited first frames
        iq_signal_full = data["iq_signal"]

        # desired output
        z_dim, x_dim, t_dim = iq_signal_full.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_full.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf_full = U[:, svd_org:] @ np.diag(Λ[svd_org:]) @ V.T[svd_org:]

        iqf_lower_limit = iqf_full[:, self.offset_to_take_frames :]
        iqf_lower_limit = iqf_lower_limit[:, :: self.interleaved]
        offset = np.random.randint(0, iqf_lower_limit.shape[-1] - self.num_frames + 1)

        iqf_lower_limit = iqf_lower_limit[:, offset : offset + self.num_frames]

        dop_signal_lower_limit = np.mean(np.abs(iqf_lower_limit) ** 2, axis=-1).reshape(
            z_dim, x_dim
        )
        dop_signal_lower_limit = torch.tensor(
            dop_signal_lower_limit, dtype=torch.float32
        )

        ##################################################################
        ##################################################################
        ##### to get from limited frames
        iq_signal = data["iq_signal"][:, :, self.offset_to_take_frames :]
        iq_signal = iq_signal[:, :, :: self.interleaved]
        iq_signal = iq_signal[:, :, offset : offset + self.num_frames]

        # desired output
        z_dim, x_dim, t_dim = iq_signal.shape
        U, Λ, V = np.linalg.svd(
            iq_signal.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal = transform(iq_signal, dop_signal)

                _, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal_with_limited_time_frames
                )

                _, dop_signal_lower_limit = transform(iq_signal, dop_signal_lower_limit)

            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            iq_signal, dop_signal = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
            )
            _, dop_signal_with_limited_time_frames = utils.standardize(
                iq_signal,
                dop_signal_with_limited_time_frames,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            _, dop_signal_lower_limit = utils.standardize(
                iq_signal,
                dop_signal_lower_limit,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        return (
            iq_signal,
            dop_signal,
            dop_signal_with_limited_time_frames,
            dop_signal_lower_limit,
            curr_input_path,
        )


class FUSdatasetOnlineDOP(torch.utils.data.Dataset):
    # this computes DOP from bin data on the fly.
    # WARNING: this is slow.

    def __init__(
        self,
        data_path: str,
        num_frames=300,
        num_frames_for_dop=300,
        delayshift=0,
        remove_top_pixels=0,
    ):
        self.data_path = data_path
        self.num_frames = num_frames
        self.num_frames_for_dop = num_frames_for_dop
        self.num_frames_full = 300
        self.num_channels = 128
        self.delayshift = delayshift
        self.remove_top_pixels = remove_top_pixels

        # make sure that the animal folder that
        # you want to load starts with "Mouse_"
        # otherwsie they wont be loaded through the dataset
        self.animal_names = os.listdir(self.data_path)
        self.animal_names = [x for x in self.animal_names if "Mouse_" in x]

        self.animal_subfolders_dict = dict()
        for animal_name in self.animal_names:
            animal_path = f"{self.data_path}/{animal_name}"
            self.animal_subfolders_dict[f"{animal_name}"] = os.listdir(animal_path)
            self.animal_subfolders_dict[f"{animal_name}"] = [
                x for x in self.animal_subfolders_dict[f"{animal_name}"] if not "." in x
            ]

        data_path_list = list()
        for animal_name in self.animal_subfolders_dict:
            for subfolder in self.animal_subfolders_dict[f"{animal_name}"]:
                curr_path = f"{self.data_path}/{animal_name}/{subfolder}"

                bin_names = os.listdir(curr_path)
                bin_names = [x for x in bin_names if ".bin" in x]

                for bin_name in bin_names:
                    bin_path = f"{curr_path}/{bin_name}"
                    data_path_list.append(bin_path)

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]
        curr_folder_path = curr_input_path.split(".")[0].split("fUS_block_")[0]

        Mouse_id = (curr_input_path.split("Mouse_")[-1]).split("/")[0]
        if Mouse_id == "1AAN":
            svd = 40
        elif Mouse_id == "1ABN":
            svd = 50
        elif Mouse_id == "2AAR":
            svd = 60
        elif Mouse_id == "2ABR":
            svd = 30
        elif Mouse_id == "A":
            svd = 60
        elif Mouse_id == "B":
            svd = 60
        else:
            raise ValueError(f"Mouse folder is not known!")

        # input
        bin_data = np.fromfile(curr_input_path, dtype="<f8")
        iq_signal_stack = bin_data.reshape(
            -1, self.num_channels, 2, self.num_frames_full, order="F"
        )
        iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

        if self.remove_top_pixels:
            iq_signal = iq_signal[self.remove_top_pixels :]

        iq_signal_for_dop_computation = iq_signal[
            :, :, self.delayshift : self.delayshift + self.num_frames_for_dop
        ]

        # desired output
        z_dim, x_dim, t_dim = iq_signal_for_dop_computation.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_for_dop_computation.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal = np.mean(np.abs(iqf) ** 2, axis=-1).reshape(z_dim, x_dim)

        # get num_frames from the input
        iq_signal = iq_signal[:, :, self.delayshift : self.delayshift + self.num_frames]

        # convert to tensor
        iq_signal_final = torch.tensor(iq_signal, dtype=torch.cfloat)
        dop_signal_final = torch.tensor(dop_signal, dtype=torch.float32)

        # iq_signal: (H, W, T) this is complex
        # dop_signal: (H, W) this is real

        return iq_signal_final, dop_signal_final, curr_input_path


class FUSdatasetOnlineDOPGeneral(torch.utils.data.Dataset):
    # this computes DOP from bin data on the fly.
    # WARNING: this is slow.

    def __init__(
        self,
        data_path: str,
        svd,
        num_frames=250,
        num_frames_for_dop=250,
        delayshift=0,
        remove_top_pixels=0,
    ):
        self.data_path = data_path
        self.svd = svd
        self.num_frames = num_frames
        self.num_frames_for_dop = num_frames_for_dop
        self.num_frames_full = 250
        self.num_channels = 128
        self.delayshift = delayshift
        self.remove_top_pixels = remove_top_pixels

        # make sure that the animal folder that
        # you want to load starts with "Mouse_"
        # otherwsie they wont be loaded through the dataset

        data_path_list = os.listdir(self.data_path)
        data_path_list = [
            f"{self.data_path}/{x}" for x in data_path_list if ".bin" in x
        ]

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # input
        bin_data = np.fromfile(curr_input_path, dtype="<f8")

        iq_signal_stack = bin_data.reshape(
            -1, self.num_channels, 2, self.num_frames_full, order="F"
        )
        iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

        if self.remove_top_pixels:
            iq_signal = iq_signal[self.remove_top_pixels :]

        iq_signal_for_dop_computation = iq_signal[
            :, :, self.delayshift : self.delayshift + self.num_frames_for_dop
        ]

        # desired output
        z_dim, x_dim, t_dim = iq_signal_for_dop_computation.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_for_dop_computation.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, self.svd :] @ np.diag(Λ[self.svd :]) @ V.T[self.svd :]
        dop_signal = np.mean(np.abs(iqf) ** 2, axis=-1).reshape(z_dim, x_dim)

        # get num_frames from the input
        iq_signal = iq_signal[:, :, self.delayshift : self.delayshift + self.num_frames]

        # convert to tensor
        iq_signal_final = torch.tensor(iq_signal, dtype=torch.cfloat)
        dop_signal_final = torch.tensor(dop_signal, dtype=torch.float32)

        # iq_signal: (H, W, T) this is complex
        # dop_signal: (H, W) this is real

        return iq_signal_final, dop_signal_final, curr_input_path


class FUSdatasetOnlineSettingIQ(torch.utils.data.Dataset):
    # this computes DOP from bin data on the fly.

    def __init__(
        self,
        data_path: str,
        num_frames=250,
        iq_signal_mode="complex",
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
        num_frames_full=250,
        num_channels=128,
        remove_top_pixels=0,
    ):
        self.data_path = data_path
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames
        self.num_frames_full = num_frames_full
        self.num_channels = num_channels
        self.remove_top_pixels = remove_top_pixels

        # make sure that the animal folder that
        # otherwsie they wont be loaded through the dataset

        data_path_list = os.listdir(self.data_path)
        data_path_list = [
            f"{self.data_path}/{x}" for x in data_path_list if ".bin" in x
        ]

        data_path_list.sort()

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # input
        bin_data = np.fromfile(curr_input_path, dtype="<f8")

        iq_signal_stack = bin_data.reshape(
            -1, self.num_channels, 2, self.num_frames_full, order="F"
        )
        iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

        if self.remove_top_pixels:
            iq_signal = iq_signal[self.remove_top_pixels :]

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, iq_signal.shape[-1] - self.num_frames
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        # get num_frames from the input
        iq_signal = iq_signal[
            :,
            :,
            self.offset_to_take_frames : self.offset_to_take_frames + self.num_frames,
        ]

        iq_signal = torch.tensor(iq_signal, dtype=torch.cfloat)

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal = transform(iq_signal)
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            dummy_dop_signal = iq_signal[:, :, 0].unsqueeze(dim=0)
            iq_signal, _ = utils.standardize(
                iq_signal,
                dummy_dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        # iq_signal: (H, W, T) this is complex

        return iq_signal, curr_input_path


class FUSdatasetOnlineSettingIQinterleaved(torch.utils.data.Dataset):
    # this computes DOP from bin data on the fly.

    def __init__(
        self,
        data_path: str,
        num_frames=250,
        iq_signal_mode="complex",
        interleaved=1,
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
        num_frames_full=250,
        num_channels=128,
        remove_top_pixels=0,
    ):
        self.data_path = data_path
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.interleaved = interleaved
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames
        self.num_frames_full = num_frames_full
        self.num_channels = num_channels
        self.remove_top_pixels = remove_top_pixels

        # make sure that the animal folder that
        # otherwsie they wont be loaded through the dataset

        data_path_list = os.listdir(self.data_path)
        data_path_list = [
            f"{self.data_path}/{x}" for x in data_path_list if ".bin" in x
        ]

        data_path_list.sort()

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # input
        bin_data = np.fromfile(curr_input_path, dtype="<f8")

        iq_signal_stack = bin_data.reshape(
            -1, self.num_channels, 2, self.num_frames_full, order="F"
        )
        iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

        if self.remove_top_pixels:
            iq_signal = iq_signal[self.remove_top_pixels :]

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, (self.total_frames - self.interleaved * self.num_frames)
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        # get num_frames from the input
        iq_signal = iq_signal[:, :, self.offset_to_take_frames :]
        iq_signal = iq_signal[:, :, :: self.interleaved]
        offset = np.random.randint(0, iq_signal.shape[-1] - self.num_frames + 1)
        iq_signal = iq_signal[:, :, offset : offset + self.num_frames]

        iq_signal = torch.tensor(iq_signal, dtype=torch.cfloat)

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal = transform(iq_signal)
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            dummy_dop_signal = iq_signal[:, :, 0].unsqueeze(dim=0)
            iq_signal, _ = utils.standardize(
                iq_signal,
                dummy_dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        # iq_signal: (H, W, T) this is complex

        return iq_signal, curr_input_path


class FUSdatasetOnlineSettingIQDOPsvd(torch.utils.data.Dataset):
    # this computes DOP from bin data on the fly.

    def __init__(
        self,
        data_path: str,
        svd,
        num_frames=250,
        iq_signal_mode="complex",
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
        num_frames_full=250,
        num_channels=128,
        total_frames=250,
        stride_for_svd=1,
        remove_top_pixels=0,
    ):
        self.data_path = data_path
        self.svd = svd
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames
        self.num_frames_full = num_frames_full
        self.num_channels = num_channels
        self.total_frames = total_frames
        self.stride_for_svd = stride_for_svd
        self.remove_top_pixels = remove_top_pixels

        # make sure that the animal folder that
        # otherwsie they wont be loaded through the dataset

        data_path_list = os.listdir(self.data_path)
        data_path_list = [
            f"{self.data_path}/{x}" for x in data_path_list if ".bin" in x
        ]

        data_path_list.sort()

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # adjusting the svd based on the number of frames
        svd = int(np.floor(self.svd * (self.num_frames / self.total_frames)))

        # input
        bin_data = np.fromfile(curr_input_path, dtype="<f8")

        iq_signal_stack = bin_data.reshape(
            -1, self.num_channels, 2, self.num_frames_full, order="F"
        )
        iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

        if self.remove_top_pixels:
            iq_signal = iq_signal[self.remove_top_pixels :]

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, iq_signal.shape[-1] - self.num_frames
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        # get num_frames from the input
        iq_signal = iq_signal[
            :,
            :,
            self.offset_to_take_frames : self.offset_to_take_frames + self.num_frames,
        ]

        ##################################################################
        ##################################################################
        ##### to get from limited frames
        # desired output
        iq_signal_for_svd = iq_signal[:, :, :: self.stride_for_svd]
        z_dim, x_dim, t_dim = iq_signal_for_svd.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_for_svd.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )
        ##################################################################

        iq_signal = torch.tensor(iq_signal, dtype=torch.cfloat)

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal_with_limited_time_frames
                )
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            iq_signal, dop_signal_with_limited_time_frames = utils.standardize(
                iq_signal,
                dop_signal_with_limited_time_frames,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        # iq_signal: (H, W, T) this is complex

        return iq_signal, dop_signal_with_limited_time_frames, curr_input_path


class FUSdatasetOnlineSettingIQDOPsvdinterleaved(torch.utils.data.Dataset):
    # this computes DOP from bin data on the fly.

    def __init__(
        self,
        data_path: str,
        svd,
        num_frames=250,
        iq_signal_mode="complex",
        interleaved=1,
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
        num_frames_full=250,
        num_channels=128,
        total_frames=250,
        remove_top_pixels=0,
    ):
        self.data_path = data_path
        self.svd = svd
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.interleaved = interleaved
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames
        self.num_frames_full = num_frames_full
        self.num_channels = num_channels
        self.total_frames = total_frames
        self.remove_top_pixels = remove_top_pixels

        # make sure that the animal folder that
        # otherwsie they wont be loaded through the dataset

        data_path_list = os.listdir(self.data_path)
        data_path_list = [
            f"{self.data_path}/{x}" for x in data_path_list if ".bin" in x
        ]

        data_path_list.sort()

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # adjusting the svd based on the number of frames
        svd = int(np.floor(self.svd * (self.num_frames / self.total_frames)))

        # input
        bin_data = np.fromfile(curr_input_path, dtype="<f8")

        iq_signal_stack = bin_data.reshape(
            -1, self.num_channels, 2, self.num_frames_full, order="F"
        )
        iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

        if self.remove_top_pixels:
            iq_signal = iq_signal[self.remove_top_pixels :]

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, (self.total_frames - self.interleaved * self.num_frames)
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        # get num_frames from the input
        iq_signal = iq_signal[:, :, self.offset_to_take_frames :]
        iq_signal = iq_signal[:, :, :: self.interleaved]
        offset = np.random.randint(0, iq_signal.shape[-1] - self.num_frames + 1)
        iq_signal = iq_signal[:, :, offset : offset + self.num_frames]

        ##################################################################
        ##################################################################
        ##### to get from limited frames
        # desired output
        iq_signal_for_svd = iq_signal
        z_dim, x_dim, t_dim = iq_signal_for_svd.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_for_svd.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )
        ##################################################################

        iq_signal = torch.tensor(iq_signal, dtype=torch.cfloat)

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal_with_limited_time_frames
                )
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            iq_signal, dop_signal_with_limited_time_frames = utils.standardize(
                iq_signal,
                dop_signal_with_limited_time_frames,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        # iq_signal: (H, W, T) this is complex

        return iq_signal, dop_signal_with_limited_time_frames, curr_input_path


class FUSdatasetOnlineSettingDOPfromSVD(torch.utils.data.Dataset):
    # this computes DOP from bin data on the fly.

    def __init__(
        self,
        data_path: str,
        svd_with_respect_to_full_frames,
        num_frames=250,
        num_frames_full=250,
        num_channels=128,
        remove_top_pixels=0,
        offset_to_take_frames=0,
    ):
        self.data_path = data_path
        self.svd_with_respect_to_full_frames = svd_with_respect_to_full_frames
        self.num_frames = num_frames
        self.num_frames_full = num_frames_full
        self.num_channels = num_channels
        self.remove_top_pixels = remove_top_pixels
        self.offset_to_take_frames = offset_to_take_frames

        self.svd = int(
            self.svd_with_respect_to_full_frames
            * (self.num_frames / self.num_frames_full)
        )

        # make sure that the animal folder that
        # otherwsie they wont be loaded through the dataset

        data_path_list = os.listdir(self.data_path)
        data_path_list = [
            f"{self.data_path}/{x}" for x in data_path_list if ".bin" in x
        ]

        data_path_list.sort()

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # input
        bin_data = np.fromfile(curr_input_path, dtype="<f8")

        iq_signal_stack = bin_data.reshape(
            -1, self.num_channels, 2, self.num_frames_full, order="F"
        )
        iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

        if self.remove_top_pixels:
            iq_signal = iq_signal[self.remove_top_pixels :]

        # get num_frames from the input
        iq_signal = iq_signal[
            :,
            :,
            self.offset_to_take_frames : self.offset_to_take_frames + self.num_frames,
        ]

        # dop output from the available frames
        z_dim, x_dim, t_dim = iq_signal.shape
        U, Λ, V = np.linalg.svd(
            iq_signal.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, self.svd :] @ np.diag(Λ[self.svd :]) @ V.T[self.svd :]
        dop_signal = np.mean(np.abs(iqf) ** 2, axis=-1).reshape(z_dim, x_dim)

        # convert to tensor
        iq_signal_final = torch.tensor(iq_signal, dtype=torch.cfloat)
        dop_signal_final = torch.tensor(dop_signal, dtype=torch.float32)

        # iq_signal: (H, W, T) this is complex
        # dop_signal: (H, W) this is real

        return iq_signal_final, dop_signal_final, curr_input_path


class FUSdatasetOnlineSettingDOPfromSVDinterleaved(torch.utils.data.Dataset):
    # this computes DOP from bin data on the fly.

    def __init__(
        self,
        data_path: str,
        svd_with_respect_to_full_frames,
        num_frames=250,
        interleaved=1,
        num_frames_full=250,
        num_channels=128,
        remove_top_pixels=0,
        offset_to_take_frames=0,
    ):
        self.data_path = data_path
        self.svd_with_respect_to_full_frames = svd_with_respect_to_full_frames
        self.num_frames = num_frames
        self.interleaved = interleaved
        self.num_frames_full = num_frames_full
        self.num_channels = num_channels
        self.remove_top_pixels = remove_top_pixels
        self.offset_to_take_frames = offset_to_take_frames

        self.svd = int(
            self.svd_with_respect_to_full_frames
            * (self.num_frames / self.num_frames_full)
        )

        # make sure that the animal folder that
        # otherwsie they wont be loaded through the dataset

        data_path_list = os.listdir(self.data_path)
        data_path_list = [
            f"{self.data_path}/{x}" for x in data_path_list if ".bin" in x
        ]

        data_path_list.sort()

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # input
        bin_data = np.fromfile(curr_input_path, dtype="<f8")

        iq_signal_stack = bin_data.reshape(
            -1, self.num_channels, 2, self.num_frames_full, order="F"
        )
        iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

        if self.remove_top_pixels:
            iq_signal = iq_signal[self.remove_top_pixels :]

        # get num_frames from the input
        iq_signal = iq_signal[:, :, self.offset_to_take_frames :]
        iq_signal = iq_signal[:, :, :: self.interleaved]
        offset = np.random.randint(0, iq_signal.shape[-1] - self.num_frames + 1)
        iq_signal = iq_signal[:, :, offset : offset + self.num_frames]

        # dop output from the available frames
        z_dim, x_dim, t_dim = iq_signal.shape
        U, Λ, V = np.linalg.svd(
            iq_signal.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, self.svd :] @ np.diag(Λ[self.svd :]) @ V.T[self.svd :]
        dop_signal = np.mean(np.abs(iqf) ** 2, axis=-1).reshape(z_dim, x_dim)

        # convert to tensor
        iq_signal_final = torch.tensor(iq_signal, dtype=torch.cfloat)
        dop_signal_final = torch.tensor(dop_signal, dtype=torch.float32)

        # iq_signal: (H, W, T) this is complex
        # dop_signal: (H, W) this is real

        return iq_signal_final, dop_signal_final, curr_input_path


class FUSdatasetOnlineSettingforTrainingDynamicTrainSize(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        svd,
        num_data,
        num_frames=250,
        iq_signal_mode="complex",
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        num_frames_full=250,
        num_channels=128,
        remove_top_pixels=0,
    ):
        self.data_path = data_path
        self.svd = svd
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.num_frames_full = num_frames_full
        self.num_channels = num_channels
        self.remove_top_pixels = remove_top_pixels

        # make sure that the animal folder that
        # otherwsie they wont be loaded through the dataset

        filename_list = os.listdir(self.data_path)
        filename_list = [x for x in filename_list if ".bin" in x]
        if num_data:
            filename_list = [
                x
                for x in filename_list
                if int(x.split("_")[-1].split(".")[0]) <= num_data
            ]
        else:
            filename_list = filename_list

        data_path_list = [f"{self.data_path}/{x}" for x in filename_list]

        data_path_list.sort()

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

        iq_signal_list = list()
        dop_signal_list = list()
        curr_input_path_list = list()
        # go over the full dataset once and get iq and dop from bin files
        for curr_input_path in self.data_path_list:
            print(f"loading {curr_input_path} for training.")

            # input
            bin_data = np.fromfile(curr_input_path, dtype="<f8")

            iq_signal_stack = bin_data.reshape(
                -1, self.num_channels, 2, self.num_frames_full, order="F"
            )
            iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

            if self.remove_top_pixels:
                iq_signal = iq_signal[self.remove_top_pixels :]

            # desired output
            z_dim, x_dim, t_dim = iq_signal.shape
            U, Λ, V = np.linalg.svd(
                iq_signal.reshape(z_dim * x_dim, t_dim),
                full_matrices=False,
            )

            iqf = U[:, self.svd :] @ np.diag(Λ[self.svd :]) @ V.T[self.svd :]
            dop_signal = np.mean(np.abs(iqf) ** 2, axis=-1).reshape(z_dim, x_dim)

            # convert to tensor
            iq_signal = torch.tensor(iq_signal, dtype=torch.cfloat)
            dop_signal = torch.tensor(dop_signal, dtype=torch.float32)

            iq_signal_list.append(iq_signal)
            dop_signal_list.append(dop_signal)
            curr_input_path_list.append(curr_input_path)

        self.iq_signal_list = torch.stack(iq_signal_list, dim=0)
        self.dop_signal_list = torch.stack(dop_signal_list, dim=0)
        self.curr_input_path_list = curr_input_path_list

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.curr_input_path_list[idx]

        iq_signal = self.iq_signal_list[idx]
        dop_signal = self.dop_signal_list[idx]

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            offset_to_take_frames = np.random.randint(
                0, iq_signal.shape[-1] - self.num_frames
            )
        else:
            # take first frames
            offset_to_take_frames = 0

        # get num_frames from the input
        iq_signal = iq_signal[
            :, :, offset_to_take_frames : offset_to_take_frames + self.num_frames
        ]

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal = transform(iq_signal, dop_signal)
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            iq_signal, dop_signal = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        iq_signal_final = iq_signal
        dop_signal_final = dop_signal

        # iq_signal: (H, W, T) this is iq_signal_mode
        # dop_signal: (H, W) this is real

        return iq_signal_final, dop_signal_final, curr_input_path


class FUSdatasetOnlineSettingforTrainingDynamicTrainSizeinterleaved(
    torch.utils.data.Dataset
):
    def __init__(
        self,
        data_path: str,
        svd,
        num_data,
        num_frames=250,
        iq_signal_mode="complex",
        interleaved=1,
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        num_frames_full=250,
        num_channels=128,
        remove_top_pixels=0,
    ):
        self.data_path = data_path
        self.svd = svd
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.interleaved = interleaved
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.num_frames_full = num_frames_full
        self.num_channels = num_channels
        self.remove_top_pixels = remove_top_pixels

        # make sure that the animal folder that
        # otherwsie they wont be loaded through the dataset

        filename_list = os.listdir(self.data_path)
        filename_list = [x for x in filename_list if ".bin" in x]
        if num_data:
            filename_list = [
                x
                for x in filename_list
                if int(x.split("_")[-1].split(".")[0]) <= num_data
            ]
        else:
            filename_list = filename_list

        data_path_list = [f"{self.data_path}/{x}" for x in filename_list]

        data_path_list.sort()

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

        iq_signal_list = list()
        dop_signal_list = list()
        curr_input_path_list = list()
        # go over the full dataset once and get iq and dop from bin files
        for curr_input_path in self.data_path_list:
            print(f"loading {curr_input_path} for training.")

            # input
            bin_data = np.fromfile(curr_input_path, dtype="<f8")

            iq_signal_stack = bin_data.reshape(
                -1, self.num_channels, 2, self.num_frames_full, order="F"
            )
            iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

            if self.remove_top_pixels:
                iq_signal = iq_signal[self.remove_top_pixels :]

            # desired output
            z_dim, x_dim, t_dim = iq_signal.shape
            U, Λ, V = np.linalg.svd(
                iq_signal.reshape(z_dim * x_dim, t_dim),
                full_matrices=False,
            )

            iqf = U[:, self.svd :] @ np.diag(Λ[self.svd :]) @ V.T[self.svd :]
            dop_signal = np.mean(np.abs(iqf) ** 2, axis=-1).reshape(z_dim, x_dim)

            # convert to tensor
            iq_signal = torch.tensor(iq_signal, dtype=torch.cfloat)
            dop_signal = torch.tensor(dop_signal, dtype=torch.float32)

            iq_signal_list.append(iq_signal)
            dop_signal_list.append(dop_signal)
            curr_input_path_list.append(curr_input_path)

        self.iq_signal_list = torch.stack(iq_signal_list, dim=0)
        self.dop_signal_list = torch.stack(dop_signal_list, dim=0)
        self.curr_input_path_list = curr_input_path_list

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.curr_input_path_list[idx]

        iq_signal = self.iq_signal_list[idx]
        dop_signal = self.dop_signal_list[idx]

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            offset_to_take_frames = np.random.randint(
                0, (self.num_frames_full - self.interleaved * self.num_frames)
            )
        else:
            # take first frames
            offset_to_take_frames = 0

        # get num_frames from the input
        iq_signal = iq_signal[:, :, offset_to_take_frames:]
        iq_signal = iq_signal[:, :, :: self.interleaved]
        offset = np.random.randint(0, iq_signal.shape[-1] - self.num_frames + 1)
        iq_signal = iq_signal[:, :, offset : offset + self.num_frames]

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal = transform(iq_signal, dop_signal)
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            iq_signal, dop_signal = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        iq_signal_final = iq_signal
        dop_signal_final = dop_signal

        # iq_signal: (H, W, T) this is iq_signal_mode
        # dop_signal: (H, W) this is real

        return iq_signal_final, dop_signal_final, curr_input_path


class FUSdatasetOnlineSettingforTrainingDynamicTrainSizeIQDOPsvd(
    torch.utils.data.Dataset
):
    def __init__(
        self,
        data_path: str,
        svd,
        svd_for_limited,
        num_data,
        num_frames=250,
        iq_signal_mode="complex",
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        num_frames_full=250,
        num_channels=128,
        total_frames=250,
        stride_for_svd=1,
        remove_top_pixels=0,
    ):
        self.data_path = data_path
        self.svd = svd
        self.svd_for_limited = svd_for_limited
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.num_frames_full = num_frames_full
        self.num_channels = num_channels
        self.total_frames = total_frames
        self.stride_for_svd = stride_for_svd
        self.remove_top_pixels = remove_top_pixels

        # make sure that the animal folder that
        # otherwsie they wont be loaded through the dataset

        filename_list = os.listdir(self.data_path)
        filename_list = [x for x in filename_list if ".bin" in x]
        if num_data:
            filename_list = [
                x
                for x in filename_list
                if int(x.split("_")[-1].split(".")[0]) <= num_data
            ]
        else:
            filename_list = filename_list

        data_path_list = [f"{self.data_path}/{x}" for x in filename_list]

        data_path_list.sort()

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

        iq_signal_list = list()
        dop_signal_list = list()
        curr_input_path_list = list()
        # go over the full dataset once and get iq and dop from bin files
        for curr_input_path in self.data_path_list:
            print(f"loading {curr_input_path} for training.")

            # input
            bin_data = np.fromfile(curr_input_path, dtype="<f8")

            iq_signal_stack = bin_data.reshape(
                -1, self.num_channels, 2, self.num_frames_full, order="F"
            )
            iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

            if self.remove_top_pixels:
                iq_signal = iq_signal[self.remove_top_pixels :]

            # desired output
            z_dim, x_dim, t_dim = iq_signal.shape
            U, Λ, V = np.linalg.svd(
                iq_signal.reshape(z_dim * x_dim, t_dim),
                full_matrices=False,
            )

            iqf = U[:, self.svd :] @ np.diag(Λ[self.svd :]) @ V.T[self.svd :]
            dop_signal = np.mean(np.abs(iqf) ** 2, axis=-1).reshape(z_dim, x_dim)

            # convert to tensor
            iq_signal = torch.tensor(iq_signal, dtype=torch.cfloat)
            dop_signal = torch.tensor(dop_signal, dtype=torch.float32)

            iq_signal_list.append(iq_signal)
            dop_signal_list.append(dop_signal)
            curr_input_path_list.append(curr_input_path)

        self.iq_signal_list = torch.stack(iq_signal_list, dim=0)
        self.dop_signal_list = torch.stack(dop_signal_list, dim=0)
        self.curr_input_path_list = curr_input_path_list

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.curr_input_path_list[idx]

        # adjusting the svd based on the number of frames
        svd = int(
            np.floor(self.svd_for_limited * (self.num_frames / self.total_frames))
        )
        iq_signal = self.iq_signal_list[idx]
        dop_signal = self.dop_signal_list[idx]

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            offset_to_take_frames = np.random.randint(
                0, iq_signal.shape[-1] - self.num_frames
            )
        else:
            # take first frames
            offset_to_take_frames = 0

        # get num_frames from the input
        iq_signal = iq_signal[
            :, :, offset_to_take_frames : offset_to_take_frames + self.num_frames
        ]

        ##################################################################
        ##################################################################
        ##### to get from limited frames
        # desired output
        iq_signal_for_svd = iq_signal[:, :, :: self.stride_for_svd]
        z_dim, x_dim, t_dim = iq_signal_for_svd.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_for_svd.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )
        ##################################################################

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal, dop_signal_with_limited_time_frames
                )
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            (
                iq_signal,
                dop_signal,
                dop_signal_with_limited_time_frames,
            ) = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
                dop_from_svd=dop_signal_with_limited_time_frames,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        # iq_signal: (H, W, T) this is iq_signal_mode
        # dop_signal: (H, W) this is real

        return (
            iq_signal,
            dop_signal,
            dop_signal_with_limited_time_frames,
            curr_input_path,
        )


class FUSdatasetOnlineSettingforTrainingDynamicTrainSizeIQDOPsvdinterleaved(
    torch.utils.data.Dataset
):
    def __init__(
        self,
        data_path: str,
        svd,
        svd_for_limited,
        num_data,
        num_frames=250,
        iq_signal_mode="complex",
        interleaved=1,
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        num_frames_full=250,
        num_channels=128,
        total_frames=250,
        remove_top_pixels=0,
    ):
        self.data_path = data_path
        self.svd = svd
        self.svd_for_limited = svd_for_limited
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.interleaved = interleaved
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.num_frames_full = num_frames_full
        self.num_channels = num_channels
        self.total_frames = total_frames
        self.remove_top_pixels = remove_top_pixels

        # make sure that the animal folder that
        # otherwsie they wont be loaded through the dataset

        filename_list = os.listdir(self.data_path)
        filename_list = [x for x in filename_list if ".bin" in x]
        if num_data:
            filename_list = [
                x
                for x in filename_list
                if int(x.split("_")[-1].split(".")[0]) <= num_data
            ]
        else:
            filename_list = filename_list

        data_path_list = [f"{self.data_path}/{x}" for x in filename_list]

        data_path_list.sort()

        # data_path_list contains the list of all bin data
        self.data_path_list = data_path_list
        self.num_data = len(self.data_path_list)

        iq_signal_list = list()
        dop_signal_list = list()
        curr_input_path_list = list()
        # go over the full dataset once and get iq and dop from bin files
        for curr_input_path in self.data_path_list:
            print(f"loading {curr_input_path} for training.")

            # input
            bin_data = np.fromfile(curr_input_path, dtype="<f8")

            iq_signal_stack = bin_data.reshape(
                -1, self.num_channels, 2, self.num_frames_full, order="F"
            )
            iq_signal = iq_signal_stack[:, :, 0] + 1j * iq_signal_stack[:, :, 1]

            if self.remove_top_pixels:
                iq_signal = iq_signal[self.remove_top_pixels :]

            # desired output
            z_dim, x_dim, t_dim = iq_signal.shape
            U, Λ, V = np.linalg.svd(
                iq_signal.reshape(z_dim * x_dim, t_dim),
                full_matrices=False,
            )

            iqf = U[:, self.svd :] @ np.diag(Λ[self.svd :]) @ V.T[self.svd :]
            dop_signal = np.mean(np.abs(iqf) ** 2, axis=-1).reshape(z_dim, x_dim)

            # convert to tensor
            iq_signal = torch.tensor(iq_signal, dtype=torch.cfloat)
            dop_signal = torch.tensor(dop_signal, dtype=torch.float32)

            iq_signal_list.append(iq_signal)
            dop_signal_list.append(dop_signal)
            curr_input_path_list.append(curr_input_path)

        self.iq_signal_list = torch.stack(iq_signal_list, dim=0)
        self.dop_signal_list = torch.stack(dop_signal_list, dim=0)
        self.curr_input_path_list = curr_input_path_list

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.curr_input_path_list[idx]

        # adjusting the svd based on the number of frames
        svd = int(
            np.floor(self.svd_for_limited * (self.num_frames / self.total_frames))
        )
        iq_signal = self.iq_signal_list[idx]
        dop_signal = self.dop_signal_list[idx]

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            offset_to_take_frames = np.random.randint(
                0, (self.total_frames - self.interleaved * self.num_frames)
            )
        else:
            # take first frames
            offset_to_take_frames = 0

        # get num_frames from the input
        iq_signal = iq_signal[:, :, offset_to_take_frames:]
        iq_signal = iq_signal[:, :, :: self.interleaved]
        offset = np.random.randint(0, iq_signal.shape[-1] - self.num_frames + 1)
        iq_signal = iq_signal[:, :, offset : offset + self.num_frames]

        ##################################################################
        ##################################################################
        ##### to get from limited frames
        # desired output
        iq_signal_for_svd = iq_signal
        z_dim, x_dim, t_dim = iq_signal_for_svd.shape
        U, Λ, V = np.linalg.svd(
            iq_signal_for_svd.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )
        ##################################################################

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal, dop_signal_with_limited_time_frames
                )
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            (
                iq_signal,
                dop_signal,
                dop_signal_with_limited_time_frames,
            ) = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
                dop_from_svd=dop_signal_with_limited_time_frames,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        # iq_signal: (H, W, T) this is iq_signal_mode
        # dop_signal: (H, W) this is real

        return (
            iq_signal,
            dop_signal,
            dop_signal_with_limited_time_frames,
            curr_input_path,
        )


class FUSdatasetDynamicsTrainSize(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        num_data,
        num_frames=300,
        iq_signal_mode="complex",
        standardization_constant=None,
        take_random_window=False,
        transform=None,
    ):
        self.data_path = data_path
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform

        filename_list = os.listdir(self.data_path)

        if num_data:
            self.filename_list = [
                x for x in filename_list if int(x.split("_")[-2]) <= num_data
            ]
        else:
            self.filename_list = filename_list

        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]

        self.num_data = len(self.data_path_list)

        if num_data:
            assert self.num_data == num_data

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # the data is a tensor
        data = torch.load(curr_input_path)

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            offset_to_take_frames = np.random.randint(
                0, data["iq_signal"].shape[-1] - self.num_frames
            )
        else:
            # take first frames
            offset_to_take_frames = 0

        iq_signal = data["iq_signal"][
            :, :, offset_to_take_frames : offset_to_take_frames + self.num_frames
        ]
        dop_signal = data["dop_signal"]

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal = transform(iq_signal, dop_signal)
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            iq_signal, dop_signal = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        #### in general
        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        #### for stack it would be
        # iq_signal: (H, W, 2T)

        return iq_signal, dop_signal, curr_input_path


class FUSdatasetDynamicsTrainSizeinterleaved(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        num_data,
        num_frames=300,
        iq_signal_mode="complex",
        interleaved=1,
        standardization_constant=None,
        take_random_window=False,
        transform=None,
    ):
        self.data_path = data_path
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.interleaved = interleaved
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform

        filename_list = os.listdir(self.data_path)

        if num_data:
            self.filename_list = [
                x for x in filename_list if int(x.split("_")[-2]) <= num_data
            ]
        else:
            self.filename_list = filename_list

        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]

        self.num_data = len(self.data_path_list)

        if num_data:
            assert self.num_data == num_data

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # the data is a tensor
        data = torch.load(curr_input_path)

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(0, self.num_frames)
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        iq_signal = data["iq_signal"][:, :, self.offset_to_take_frames :]
        iq_signal = iq_signal[:, :, :: self.interleaved]
        offset = np.random.randint(0, iq_signal.shape[-1] - self.num_frames + 1)
        iq_signal = iq_signal[:, :, offset : offset + self.num_frames]

        dop_signal = data["dop_signal"]

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal = transform(iq_signal, dop_signal)
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            iq_signal, dop_signal = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        #### in general
        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        #### for stack it would be
        # iq_signal: (H, W, 2T)

        return iq_signal, dop_signal, curr_input_path


class FUSdatasetDynamicsTrainSizeIQDOPsvd(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        num_data,
        svd,
        num_frames=300,
        iq_signal_mode="complex",
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
        total_frames=250,
    ):
        self.data_path = data_path
        self.svd = svd
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames
        self.total_frames = total_frames

        filename_list = os.listdir(self.data_path)

        if num_data:
            self.filename_list = [
                x for x in filename_list if int(x.split("_")[-2]) <= num_data
            ]
        else:
            self.filename_list = filename_list

        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]

        self.num_data = len(self.data_path_list)

        if num_data:
            assert self.num_data == num_data

        iq_signal_list = list()
        dop_signal_list = list()
        curr_input_path_list = list()
        # go over the full dataset once and get iq and dop from bin files
        for curr_input_path in self.data_path_list:
            print(f"loading {curr_input_path} for training.")

            data = torch.load(curr_input_path)

            iq_signal_list.append(data["iq_signal"])
            dop_signal_list.append(data["dop_signal"])
            curr_input_path_list.append(curr_input_path)

        self.iq_signal_list = torch.stack(iq_signal_list, dim=0)
        self.dop_signal_list = torch.stack(dop_signal_list, dim=0)
        self.curr_input_path_list = curr_input_path_list

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # adjusting the svd based on the number of frames
        svd = int(np.floor(self.svd * (self.num_frames / self.total_frames)))

        # the data is a tensor
        data = dict()
        data["iq_signal"] = self.iq_signal_list[idx]
        data["dop_signal"] = self.dop_signal_list[idx]

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, data["iq_signal"].shape[-1] - self.num_frames
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        iq_signal = data["iq_signal"][
            :,
            :,
            self.offset_to_take_frames : self.offset_to_take_frames + self.num_frames,
        ]
        dop_signal = data["dop_signal"]

        ##################################################################
        ##################################################################
        ##### to get from limited frames
        # desired output
        z_dim, x_dim, t_dim = iq_signal.shape
        U, Λ, V = np.linalg.svd(
            iq_signal.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )
        ##################################################################

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal, dop_signal_with_limited_time_frames
                )
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            (
                iq_signal,
                dop_signal,
                dop_signal_with_limited_time_frames,
            ) = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
                dop_from_svd=dop_signal_with_limited_time_frames,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        #### in general
        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        #### for stack it would be
        # iq_signal: (H, W, 2T)

        return (
            iq_signal,
            dop_signal,
            dop_signal_with_limited_time_frames,
            curr_input_path,
        )


class FUSdatasetDynamicsTrainSizeIQDOPsvdinterleaved(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        num_data,
        svd,
        num_frames=300,
        iq_signal_mode="complex",
        interleaved=1,
        standardization_constant=None,
        take_random_window=False,
        transform=None,
        offset_to_take_frames=0,
        total_frames=250,
    ):
        self.data_path = data_path
        self.svd = svd
        self.num_frames = num_frames
        self.iq_signal_mode = iq_signal_mode  # real, abs, complex
        self.interleaved = interleaved
        self.standardization_constant = standardization_constant
        self.take_random_window = take_random_window
        self.transform = transform
        self.offset_to_take_frames = offset_to_take_frames
        self.total_frames = total_frames

        filename_list = os.listdir(self.data_path)

        if num_data:
            self.filename_list = [
                x for x in filename_list if int(x.split("_")[-2]) <= num_data
            ]
        else:
            self.filename_list = filename_list

        self.data_path_list = [
            f"{self.data_path}/{x}" for x in self.filename_list if ".pt" in x
        ]

        self.num_data = len(self.data_path_list)

        if num_data:
            assert self.num_data == num_data

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int):
        curr_input_path = self.data_path_list[idx]

        # adjusting the svd based on the number of frames
        svd = int(np.floor(self.svd * (self.num_frames / self.total_frames)))

        # the data is a tensor
        data = torch.load(curr_input_path)

        # get num_frames from the input
        if self.take_random_window:
            # have an offset to take (mainly used for training)
            self.offset_to_take_frames = np.random.randint(
                0, (self.total_frames - self.interleaved * self.num_frames)
            )
        else:
            if not self.offset_to_take_frames:
                # take first frames
                self.offset_to_take_frames = 0

        iq_signal = data["iq_signal"][:, :, self.offset_to_take_frames :]
        iq_signal = iq_signal[:, :, :: self.interleaved]
        offset = np.random.randint(0, iq_signal.shape[-1] - self.num_frames + 1)
        iq_signal = iq_signal[:, :, offset : offset + self.num_frames]

        dop_signal = data["dop_signal"]
        ##################################################################
        ##################################################################
        ##### to get from limited frames
        # desired output
        z_dim, x_dim, t_dim = iq_signal.shape
        U, Λ, V = np.linalg.svd(
            iq_signal.reshape(z_dim * x_dim, t_dim),
            full_matrices=False,
        )

        iqf = U[:, svd:] @ np.diag(Λ[svd:]) @ V.T[svd:]
        dop_signal_with_limited_time_frames = np.mean(
            np.abs(iqf) ** 2, axis=-1
        ).reshape(z_dim, x_dim)
        dop_signal_with_limited_time_frames = torch.tensor(
            dop_signal_with_limited_time_frames, dtype=torch.float32
        )
        ##################################################################

        if self.iq_signal_mode == "real":
            iq_signal = torch.real(iq_signal)
        elif self.iq_signal_mode == "abs":
            iq_signal = torch.abs(iq_signal)
        elif self.iq_signal_mode == "stack":
            iq_signal = torch.cat(
                [iq_signal.real, iq_signal.imag], dim=-1
            )  # along the time dimension
        elif self.iq_signal_mode == "complex":
            pass

        # perform tansform
        if self.transform:
            iq_signal = iq_signal.permute(2, 0, 1)
            for transform in self.transform:
                iq_signal, dop_signal, dop_signal_with_limited_time_frames = transform(
                    iq_signal, dop_signal, dop_signal_with_limited_time_frames
                )
            iq_signal = iq_signal.permute(1, 2, 0)

        # standardization is computed after the zeropadding for images to have all the same dimension
        # so apply standardization_constant after the transform
        # do standardization
        if self.standardization_constant:
            # move the time dim into 2.

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(2, 0, 1)

            (
                iq_signal,
                dop_signal,
                dop_signal_with_limited_time_frames,
            ) = utils.standardize(
                iq_signal,
                dop_signal,
                self.standardization_constant,
                self.iq_signal_mode,
                dop_from_svd=dop_signal_with_limited_time_frames,
            )

            if self.standardization_constant["iq_mean"].dim() == 2:
                iq_signal = iq_signal.permute(1, 2, 0)

        #### in general
        # iq_signal: (H, W, T)
        # dop_signal: (H, W) this is real

        #### for stack it would be
        # iq_signal: (H, W, 2T)

        return (
            iq_signal,
            dop_signal,
            dop_signal_with_limited_time_frames,
            curr_input_path,
        )
