"""
Copyright (c) 2025 Bahareh Tolooshams

model

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import torch.nn.functional as F


class SpectralConv1ddim(torch.nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super(SpectralConv1ddim, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        self.scale = 1 / (in_ch * out_ch)
        self.weights1 = torch.nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float)
        )

    def compl_mul1d(self, a, b):
        # (B, M, in_ch, H, W), (in_ch, out_ch, M) -> (B, M, out_channel, H, W)
        return torch.einsum("bmihw,iom->bmohw", a, b)

    def forward(self, x, dur_increased_factor=None):
        B, T, C, H, W = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        with torch.autocast(device_type="cuda", enabled=False):
            if dur_increased_factor:
                raise NotImplementedError(
                    "dur_increased_factor is not implemented to real!"
                )
            else:
                x_ft = torch.fft.rfftn(x.float(), dim=[1])
            # Multiply relevant Fourier modes
            out_ft = self.compl_mul1d(
                x_ft[:, : self.modes1], torch.view_as_complex(self.weights1)
            )
            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=[T], dim=[1])
        return x


class ComplexSpectralConv1ddim(torch.nn.Module):
    def __init__(self, in_ch, out_ch, modes1, skip_dc=False, init_conj=False):
        super(ComplexSpectralConv1ddim, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip_dc = skip_dc
        self.init_conj = init_conj
        if self.skip_dc:
            self.modes1 = modes1 - 1
            self.modes1_neg = modes1 - 1  # neg freq does not have dc 0.
        else:
            self.modes1 = modes1
            self.modes1_neg = modes1 - 1  # neg freq does not have dc 0.

        self.scale = 1 / (in_ch * out_ch)
        self.weights1 = torch.nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float)
        )
        self.weights1_neg = torch.nn.Parameter(
            self.scale
            * torch.rand(in_ch, out_ch, self.modes1_neg, 2, dtype=torch.float)
        )

        if self.init_conj:
            if self.skip_dc:
                self.weights1_neg.data = torch.conj(
                    torch.flip(self.weights1.data.clone(), dims=[-1])
                )
            else:
                self.weights1_neg.data = torch.conj(
                    torch.flip(self.weights1.data[:, :, 1:].clone(), dims=[-1])
                )

    def compl_mul1d(self, a, b):
        # (B, M, in_ch, H, W), (in_ch, out_ch, M) -> (B, M, out_channel, H, W)
        return torch.einsum("bmihw,iom->bmohw", a, b)

    def forward(self, x, dur_increased_factor=None):
        B, T, C, H, W = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        with torch.autocast(device_type="cuda", enabled=False):
            x_ft = torch.fft.fftn(x.cfloat(), s=[T], dim=[1])
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                B,
                x_ft.size(1),
                self.out_ch,
                H,
                W,
                dtype=torch.cfloat,
                device=x_ft.device,
            )
            # pos freq

            if dur_increased_factor:
                n_modes = int(self.weights1.shape[-2] * dur_increased_factor)
                w1 = torch.nn.functional.interpolate(self.weights1, size=(n_modes, 2))
                if dur_increased_factor == 2:
                    a = 1
                    w1 = w1[:, :, :-a, :]
                elif dur_increased_factor == 4:
                    a = 3
                    w1 = w1[:, :, :-a, :]
                else:
                    a = 0
                    print("not implemented")

                w1 = torch.view_as_complex(w1)

                n_modes = int(self.weights1_neg.shape[-2] * dur_increased_factor)
                w1n = torch.nn.functional.interpolate(
                    self.weights1_neg, size=(n_modes, 2)
                )
                w1n = torch.view_as_complex(w1n)
            else:
                w1 = torch.view_as_complex(self.weights1)
                w1n = torch.view_as_complex(self.weights1_neg)

                a = 0
                dur_increased_factor = 1

            if self.skip_dc:
                out_ft[
                    :, 1 : 1 + (self.modes1 * dur_increased_factor)
                ] = self.compl_mul1d(
                    x_ft[:, 1 : 1 + (self.modes1 * dur_increased_factor)], w1
                )
            else:
                out_ft[
                    :, : (self.modes1 * dur_increased_factor) - a
                ] = self.compl_mul1d(
                    x_ft[:, : (self.modes1 * dur_increased_factor) - a], w1
                )
            out_ft[:, -(self.modes1_neg * dur_increased_factor) :] = self.compl_mul1d(
                x_ft[:, -(self.modes1_neg * dur_increased_factor) :], w1n
            )
            # Return to physical space
            x = torch.fft.ifftn(out_ft, s=[T], dim=[1])
        return x


class SpectralConv1ddimhighfreq(torch.nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super(SpectralConv1ddimhighfreq, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        self.scale = 1 / (in_ch * out_ch)
        self.weights1 = torch.nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float)
        )

    def compl_mul1d(self, a, b):
        # (B, M, in_ch, H, W), (in_ch, out_ch, M) -> (B, M, out_channel, H, W)
        return torch.einsum("bmihw,iom->bmohw", a, b)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        with torch.autocast(device_type="cuda", enabled=False):
            x_ft = torch.fft.rfftn(x.float(), dim=[1])
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                B,
                x_ft.size(1),
                self.out_ch,
                H,
                W,
                dtype=torch.cfloat,
                device=x_ft.device,
            )
            ##############################
            ##############################
            ##############################
            ##### modes are in high freq here
            out_ft[:, -self.modes1 :] = self.compl_mul1d(
                x_ft[:, -self.modes1 :], torch.view_as_complex(self.weights1)
            )
            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=[T], dim=[1])
        return x


class ComplexConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(ComplexConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(
            2 * in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def apply_complex(self, f, input, dtype=torch.float):
        input_channels = int(input.shape[1] / 2)
        input_real = input[:, :input_channels]
        input_imag = input[:, input_channels:]

        out_real = (f(torch.cat([input_real, -input_imag], dim=1))).type(dtype)
        out_imag = (f(torch.cat([input_imag, input_real], dim=1))).type(dtype)

        return torch.cat([out_real, out_imag], dim=1)

    def forward(self, x):
        return self.apply_complex(self.conv, x)


class ComplexConvTranspose2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(
            2 * in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
        )

    def apply_complex(self, f, input, dtype=torch.float):
        input_channels = int(input.shape[1] / 2)
        input_real = input[:, :input_channels]
        input_imag = input[:, input_channels:]

        out_real = (f(torch.cat([input_real, -input_imag], dim=1))).type(dtype)
        out_imag = (f(torch.cat([input_imag, input_real], dim=1))).type(dtype)

        return torch.cat([out_real, out_imag], dim=1)

    def forward(self, x):
        return self.apply_complex(self.conv, x)


class VARSfUSI(torch.nn.ModuleDict):
    def __init__(self, params):
        super().__init__()

        class ComplexConv(torch.nn.Sequential):
            def __init__(self, idim, odim):
                super().__init__()

                if params.model_params.residual:
                    self.conv_matching_residual_dim = ComplexConv2d(
                        idim, odim, 1, padding="same"
                    )
                    block_idim = odim
                else:
                    block_idim = idim

                self.block = torch.nn.Sequential(
                    ComplexConv2d(block_idim, odim, 3, padding="same"),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Dropout2d(params.model_params.dropout),
                    ComplexConv2d(odim, odim, 3, padding="same"),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Dropout2d(params.model_params.dropout),
                )

            def forward(self, x):
                if params.model_params.residual:
                    x = self.conv_matching_residual_dim(x)

                y = self.block(x)

                if params.model_params.residual:
                    y = y + x

                return y

        class Conv(torch.nn.Sequential):
            def __init__(self, idim, odim):
                super().__init__()

                if params.model_params.residual:
                    self.conv_matching_residual_dim = torch.nn.Conv2d(
                        idim, odim, 1, padding="same"
                    )
                    block_idim = odim
                else:
                    block_idim = idim

                self.block = torch.nn.Sequential(
                    torch.nn.Conv2d(block_idim, odim, 3, padding="same"),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(params.model_params.dropout),
                    torch.nn.Conv2d(odim, odim, 3, padding="same"),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(params.model_params.dropout),
                )

            def forward(self, x):
                if params.model_params.residual:
                    x = self.conv_matching_residual_dim(x)

                y = self.block(x)

                if params.model_params.residual:
                    y += x

                return y

        def ComplexTranspose(idim: int, odim: int):
            kwargs = dict(kernel_size=3, stride=2, padding=1, output_padding=1)
            return torch.nn.Sequential(
                ComplexConvTranspose2d(idim, odim, **kwargs), torch.nn.ReLU()
            )

        self.mean_or_max = params.model_params.mean_or_max
        self.out_method = params.model_params.out_method

        idim = 1

        self["input"] = ComplexConv(idim, params.model_params.num_channels[0])
        self["input_svd"] = Conv(idim, 2 * params.model_params.num_channels[0])
        self["output"] = ComplexConv2d(
            params.model_params.num_channels[0],
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )

        self["encode"] = torch.nn.ModuleList(
            map(
                ComplexConv,
                params.model_params.num_channels[:-1],
                params.model_params.num_channels[1:],
            )
        )
        self["decode"] = torch.nn.ModuleList(
            map(
                ComplexConv,
                params.model_params.num_channels[1:],
                params.model_params.num_channels[:-1],
            )
        )

        self["T"] = torch.nn.ModuleList(
            map(
                ComplexTranspose,
                params.model_params.num_channels[1:],
                params.model_params.num_channels[:-1],
            )
        )

        self.specconv_time = torch.nn.ModuleList([])
        self.num_layers = len(params.model_params.num_channels) * 2 - 1
        self.fno_act = torch.nn.LeakyReLU()
        self.high_freq = params.model_params.high_freq

        for layer_idx in range(self.num_layers):
            if layer_idx < (self.num_layers / 2):
                in_ch = params.model_params.num_channels[layer_idx]
                out_ch = params.model_params.num_channels[layer_idx]
            else:
                in_ch = params.model_params.num_channels[
                    self.num_layers - layer_idx - 1
                ]
                out_ch = params.model_params.num_channels[
                    self.num_layers - layer_idx - 1
                ]
            if self.high_freq:
                self.specconv_time.append(
                    ComplexSpectralConv1ddimhighfreq(
                        in_ch, out_ch, params.model_params.fno_modes
                    )
                )
            else:
                self.specconv_time.append(
                    ComplexSpectralConv1ddim(
                        in_ch,
                        out_ch,
                        params.model_params.fno_modes,
                        params.model_params.skip_dc,
                        params.model_params.init_conj,
                    )
                )

        self.num_frames = params.num_frames

        if self.out_method == "realplusimag":
            self.complex_to_onemap_conv = torch.nn.Conv3d(
                2, 1, kernel_size=1, bias=True
            )

    def do_padding(self, x, x_svd, divisible=16):
        N, c, H, W = x.shape

        # make sure the H,W is divisible by divisible (2^4) (unet depth)
        h_pad = divisible - H % divisible
        w_pad = divisible - W % divisible
        if h_pad == divisible:
            h_pad = 0
        if w_pad == divisible:
            w_pad = 0

        padding = [w_pad // 2, h_pad // 2]

        # pad both sides
        unpad_list = list()
        for p in padding[::-1]:
            if p == 0:
                unpad_amount_pos = None
                unpad_amount_neg = None
            else:
                unpad_amount_pos = p
                unpad_amount_neg = -p
            unpad_list.append(slice(unpad_amount_pos, unpad_amount_neg, None))
        unpad_indices = (Ellipsis,) + tuple(unpad_list)

        padding = [i for p in padding for i in (p, p)]

        x_padded = F.pad(x, padding, mode="constant")
        x_svd_padded = F.pad(x_svd, padding, mode="constant")

        out_put_shape = x_padded.shape[2:]

        self._unpad_indices = unpad_indices

        return x_padded, x_svd_padded

    def unpad(self, x):
        """Remove the padding from padding inputs"""
        unpad_indices = self._unpad_indices
        return x[unpad_indices]

    def do_specconv_time(self, x, layer_ctr, N, nt, dur_increased_factor=None):
        # The channels are real and then imag in x
        _, c, H, W = x.shape
        x = x.reshape(N, nt, c, H, W)  # (batch, nt, c, nx, nz)

        x_channels = int(x.shape[2] / 2)
        x_real = x[:, :, :x_channels]
        x_imag = x[:, :, x_channels:]

        x_complex = torch.complex(x_real, x_imag).to(torch.cfloat)
        no_x_complex = self.specconv_time[layer_ctr](
            x_complex, dur_increased_factor
        )  # (batch, nt, c, nx, nz)
        no_x = torch.cat([no_x_complex.real, no_x_complex.imag], dim=2)

        x = self.fno_act(no_x) + x
        x = x.reshape(N * nt, c, H, W)  # (batch * nt, c, nx, nz)

        return x

    def forward(self, x, x_svd, dur_increased_factor=None):
        # (batch, nx, nz, 2 nt) or complex as (batch, nx, nz, nt)
        # x_svd is (batch, 1, nx, nz)

        if torch.is_complex(x):
            x = torch.stack([x.real, x.imag], dim=1).to(
                torch.float
            )  # (batch, 2, nx, nz, nt)
        else:
            x_time_ch = int(x.shape[-1] / 2)
            x_real = x[:, :, :, :x_time_ch]
            x_imag = x[:, :, :, x_time_ch:]
            x = torch.stack([x_real, x_imag], dim=1).to(
                torch.float
            )  # (batch, 2, nx, nz, nt)

        x = x.permute(0, 4, 1, 2, 3)  # (batch, nt, 2, nx, nz)

        N, nt, c, H, W = x.shape

        x = x.reshape(N * nt, c, H, W)  # (batch * T, 2, nx, nz)

        ########################################################

        x, x_svd = self.do_padding(x, x_svd)

        x = self["input"](x)
        x_svd = self["input_svd"](x_svd)
        x = x + x_svd
        x = self.do_specconv_time(x, 0, N, nt, dur_increased_factor)

        skip = [x]

        layer_ctr = 0
        for conv in self["encode"]:
            layer_ctr += 1
            x = F.max_pool2d(x, 2, 2)
            x = conv(x)

            x = self.do_specconv_time(x, layer_ctr, N, nt, dur_increased_factor)

            skip.append(x)

        T = reversed(self["T"])
        Conv = reversed(self["decode"])

        for t, conv in zip(T, Conv):
            layer_ctr += 1

            x = t(x), skip.pop(-2)
            x = conv(torch.concat(x, 1))

            x = self.do_specconv_time(x, layer_ctr, N, nt, dur_increased_factor)

        x = self["output"](x)

        x = self.unpad(x)

        ########################################################
        _, c, H, W = x.shape
        x = x.view(N, nt, c, H, W)  # (batch, nt, 2, nx, nz)

        if self.out_method == "realplusimag":
            # x is complex with (batch, nt, 2, nx, nz)

            x = x.permute(0, 2, 1, 3, 4)  # (batch, 2, nt, nx, nz)
            x_out = self.complex_to_onemap_conv(x)  # (batch, 1, nt, nx, nz)
            if self.mean_or_max == "max":
                x = torch.max(x_out, dim=(2))[0]  # output is (batch, 1, nx, nz)
            elif self.mean_or_max == "mean":
                x = torch.mean(x_out, dim=(2))  # output is (batch, 1, nx, nz)

        return x


class VARSfUSIwoSG(torch.nn.ModuleDict):
    def __init__(self, params):
        super().__init__()

        class ComplexConv(torch.nn.Sequential):
            def __init__(self, idim, odim):
                super().__init__()

                if params.model_params.residual:
                    self.conv_matching_residual_dim = ComplexConv2d(
                        idim, odim, 1, padding="same"
                    )
                    block_idim = odim
                else:
                    block_idim = idim

                self.block = torch.nn.Sequential(
                    ComplexConv2d(block_idim, odim, 3, padding="same"),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Dropout2d(params.model_params.dropout),
                    ComplexConv2d(odim, odim, 3, padding="same"),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Dropout2d(params.model_params.dropout),
                )

            def forward(self, x):
                if params.model_params.residual:
                    x = self.conv_matching_residual_dim(x)

                y = self.block(x)

                if params.model_params.residual:
                    y = y + x

                return y

        def ComplexTranspose(idim: int, odim: int):
            kwargs = dict(kernel_size=3, stride=2, padding=1, output_padding=1)
            return torch.nn.Sequential(
                ComplexConvTranspose2d(idim, odim, **kwargs), torch.nn.ReLU()
            )

        self.mean_or_max = params.model_params.mean_or_max
        self.out_method = params.model_params.out_method

        idim = 1

        self["input"] = ComplexConv(idim, params.model_params.num_channels[0])
        self["output"] = ComplexConv2d(
            params.model_params.num_channels[0],
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )

        self["encode"] = torch.nn.ModuleList(
            map(
                ComplexConv,
                params.model_params.num_channels[:-1],
                params.model_params.num_channels[1:],
            )
        )
        self["decode"] = torch.nn.ModuleList(
            map(
                ComplexConv,
                params.model_params.num_channels[1:],
                params.model_params.num_channels[:-1],
            )
        )

        self["T"] = torch.nn.ModuleList(
            map(
                ComplexTranspose,
                params.model_params.num_channels[1:],
                params.model_params.num_channels[:-1],
            )
        )

        self.specconv_time = torch.nn.ModuleList([])
        self.num_layers = len(params.model_params.num_channels) * 2 - 1
        self.fno_act = torch.nn.LeakyReLU()
        self.high_freq = params.model_params.high_freq

        for layer_idx in range(self.num_layers):
            if layer_idx < (self.num_layers / 2):
                in_ch = params.model_params.num_channels[layer_idx]
                out_ch = params.model_params.num_channels[layer_idx]
            else:
                in_ch = params.model_params.num_channels[
                    self.num_layers - layer_idx - 1
                ]
                out_ch = params.model_params.num_channels[
                    self.num_layers - layer_idx - 1
                ]
            if self.high_freq:
                self.specconv_time.append(
                    ComplexSpectralConv1ddimhighfreq(
                        in_ch, out_ch, params.model_params.fno_modes
                    )
                )
            else:
                self.specconv_time.append(
                    ComplexSpectralConv1ddim(
                        in_ch,
                        out_ch,
                        params.model_params.fno_modes,
                        params.model_params.skip_dc,
                        params.model_params.init_conj,
                    )
                )

        self.num_frames = params.num_frames

        if self.out_method == "realplusimag":
            self.complex_to_onemap_conv = torch.nn.Conv3d(
                2, 1, kernel_size=1, bias=True
            )

    def do_padding(self, x, divisible=16):
        N, c, H, W = x.shape

        # make sure the H,W is divisible by divisible (2^4) (unet depth)
        h_pad = divisible - H % divisible
        w_pad = divisible - W % divisible
        if h_pad == divisible:
            h_pad = 0
        if w_pad == divisible:
            w_pad = 0

        padding = [w_pad // 2, h_pad // 2]

        # pad both sides
        unpad_list = list()
        for p in padding[::-1]:
            if p == 0:
                unpad_amount_pos = None
                unpad_amount_neg = None
            else:
                unpad_amount_pos = p
                unpad_amount_neg = -p
            unpad_list.append(slice(unpad_amount_pos, unpad_amount_neg, None))
        unpad_indices = (Ellipsis,) + tuple(unpad_list)

        padding = [i for p in padding for i in (p, p)]

        x_padded = F.pad(x, padding, mode="constant")

        out_put_shape = x_padded.shape[2:]

        self._unpad_indices = unpad_indices

        return x_padded

    def unpad(self, x):
        """Remove the padding from padding inputs"""
        unpad_indices = self._unpad_indices
        return x[unpad_indices]

    def do_specconv_time(self, x, layer_ctr, N, nt, dur_increased_factor=None):
        # The channels are real and then imag in x
        _, c, H, W = x.shape
        x = x.reshape(N, nt, c, H, W)  # (batch, nt, c, nx, nz)

        x_channels = int(x.shape[2] / 2)
        x_real = x[:, :, :x_channels]
        x_imag = x[:, :, x_channels:]

        x_complex = torch.complex(x_real, x_imag).to(torch.cfloat)
        no_x_complex = self.specconv_time[layer_ctr](
            x_complex, dur_increased_factor
        )  # (batch, nt, c, nx, nz)
        no_x = torch.cat([no_x_complex.real, no_x_complex.imag], dim=2)

        x = self.fno_act(no_x) + x
        x = x.reshape(N * nt, c, H, W)  # (batch * nt, c, nx, nz)

        return x

    def forward(self, x, dur_increased_factor=None):
        # (batch, nx, nz, 2 nt) or complex as (batch, nx, nz, nt)

        if torch.is_complex(x):
            x = torch.stack([x.real, x.imag], dim=1).to(
                torch.float
            )  # (batch, 2, nx, nz, nt)
        else:
            x_time_ch = int(x.shape[-1] / 2)
            x_real = x[:, :, :, :x_time_ch]
            x_imag = x[:, :, :, x_time_ch:]
            x = torch.stack([x_real, x_imag], dim=1).to(
                torch.float
            )  # (batch, 2, nx, nz, nt)

        x = x.permute(0, 4, 1, 2, 3)  # (batch, nt, 2, nx, nz)

        N, nt, c, H, W = x.shape

        x = x.reshape(N * nt, c, H, W)  # (batch * T, 2, nx, nz)

        ########################################################

        x = self.do_padding(x)

        x = self["input"](x)
        x = self.do_specconv_time(x, 0, N, nt, dur_increased_factor)

        skip = [x]

        layer_ctr = 0
        for conv in self["encode"]:
            layer_ctr += 1
            x = F.max_pool2d(x, 2, 2)
            x = conv(x)

            x = self.do_specconv_time(x, layer_ctr, N, nt, dur_increased_factor)

            skip.append(x)

        T = reversed(self["T"])
        Conv = reversed(self["decode"])

        for t, conv in zip(T, Conv):
            layer_ctr += 1

            x = t(x), skip.pop(-2)
            x = conv(torch.concat(x, 1))

            x = self.do_specconv_time(x, layer_ctr, N, nt, dur_increased_factor)

        x = self["output"](x)

        x = self.unpad(x)

        ########################################################
        _, c, H, W = x.shape
        x = x.view(N, nt, c, H, W)  # (batch, nt, 2, nx, nz)

        if self.out_method == "realplusimag":
            # x is complex with (batch, nt, 2, nx, nz)

            x = x.permute(0, 2, 1, 3, 4)  # (batch, 2, nt, nx, nz)
            x_out = self.complex_to_onemap_conv(x)  # (batch, 1, nt, nx, nz)
            if self.mean_or_max == "max":
                x = torch.max(x_out, dim=(2))[0]  # output is (batch, 1, nx, nz)
            elif self.mean_or_max == "mean":
                x = torch.mean(x_out, dim=(2))  # output is (batch, 1, nx, nz)

        return x


class VARSfUSIrealwoSG(torch.nn.ModuleDict):
    def __init__(self, params):
        super().__init__()

        class Conv(torch.nn.Sequential):
            def __init__(self, idim, odim):
                super().__init__()

                if params.model_params.residual:
                    self.conv_matching_residual_dim = torch.nn.Conv2d(
                        idim, odim, 1, padding="same"
                    )
                    block_idim = odim
                else:
                    block_idim = idim

                self.block = torch.nn.Sequential(
                    torch.nn.Conv2d(block_idim, odim, 3, padding="same"),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(params.model_params.dropout),
                    torch.nn.Conv2d(odim, odim, 3, padding="same"),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(params.model_params.dropout),
                )

            def forward(self, x):
                if params.model_params.residual:
                    x = self.conv_matching_residual_dim(x)

                y = self.block(x)

                if params.model_params.residual:
                    y += x

                return y

        def Transpose(idim: int, odim: int):
            kwargs = dict(kernel_size=3, stride=2, padding=1, output_padding=1)
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(idim, odim, **kwargs), torch.nn.ReLU()
            )

        self.mean_or_max = params.model_params.mean_or_max

        idim = 1

        self["input"] = Conv(idim, params.model_params.num_channels[0])
        self["output"] = torch.nn.Conv2d(
            params.model_params.num_channels[0],
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )

        self["encode"] = torch.nn.ModuleList(
            map(
                Conv,
                params.model_params.num_channels[:-1],
                params.model_params.num_channels[1:],
            )
        )
        self["decode"] = torch.nn.ModuleList(
            map(
                Conv,
                params.model_params.num_channels[1:],
                params.model_params.num_channels[:-1],
            )
        )

        self["T"] = torch.nn.ModuleList(
            map(
                Transpose,
                params.model_params.num_channels[1:],
                params.model_params.num_channels[:-1],
            )
        )

        self.specconv_time = torch.nn.ModuleList([])
        self.num_layers = len(params.model_params.num_channels) * 2 - 1
        self.fno_act = torch.nn.LeakyReLU()
        self.high_freq = params.model_params.high_freq

        for layer_idx in range(self.num_layers):
            if layer_idx < (self.num_layers / 2):
                in_ch = params.model_params.num_channels[layer_idx]
                out_ch = params.model_params.num_channels[layer_idx]
            else:
                in_ch = params.model_params.num_channels[
                    self.num_layers - layer_idx - 1
                ]
                out_ch = params.model_params.num_channels[
                    self.num_layers - layer_idx - 1
                ]
            if self.high_freq:
                self.specconv_time.append(
                    SpectralConv1ddimhighfreq(
                        in_ch, out_ch, params.model_params.fno_modes
                    )
                )
            else:
                self.specconv_time.append(
                    SpectralConv1ddim(in_ch, out_ch, params.model_params.fno_modes)
                )

        self.num_frames = params.num_frames

    def do_padding(self, x, divisible=16):
        N, c, H, W = x.shape

        # make sure the H,W is divisible by divisible (2^4) (unet depth)
        h_pad = divisible - H % divisible
        w_pad = divisible - W % divisible
        if h_pad == divisible:
            h_pad = 0
        if w_pad == divisible:
            w_pad = 0

        padding = [w_pad // 2, h_pad // 2]

        # pad both sides
        unpad_list = list()
        for p in padding[::-1]:
            if p == 0:
                unpad_amount_pos = None
                unpad_amount_neg = None
            else:
                unpad_amount_pos = p
                unpad_amount_neg = -p
            unpad_list.append(slice(unpad_amount_pos, unpad_amount_neg, None))
        unpad_indices = (Ellipsis,) + tuple(unpad_list)

        padding = [i for p in padding for i in (p, p)]

        x_padded = F.pad(x, padding, mode="constant")

        out_put_shape = x_padded.shape[2:]

        self._unpad_indices = unpad_indices

        return x_padded

    def unpad(self, x):
        """Remove the padding from padding inputs"""
        unpad_indices = self._unpad_indices
        return x[unpad_indices]

    def do_specconv_time(self, x, layer_ctr, N, nt, dur_increased_factor=None):
        _, c, H, W = x.shape
        x = x.view(N, nt, c, H, W)  # (batch, nt, c, nx, nz)
        no_x = self.specconv_time[layer_ctr](x, dur_increased_factor)
        x = self.fno_act(no_x) + x
        x = x.reshape(N * nt, c, H, W)  # (batch * nt, c, nx, nz)

        return x

    def forward(self, x, dur_increased_factor=None):
        ###### this is FNO
        N, H, W, T = x.shape

        x = x.unsqueeze(dim=1)  # (batch, 1, nx, nz, nt)

        x = x.permute(0, 4, 1, 2, 3)  # (batch, nt, 1, nx, nz)

        ###### this is UNet
        N, nt, c, H, W = x.shape

        x = x.reshape(N * nt, c, H, W)  # (batch * T, c, nx, nz)

        ########################################################

        x = self.do_padding(x)

        x = self["input"](x)
        x = self.do_specconv_time(x, 0, N, nt, dur_increased_factor)

        skip = [x]

        layer_ctr = 0
        for conv in self["encode"]:
            layer_ctr += 1
            x = F.max_pool2d(x, 2, 2)
            x = conv(x)

            x = self.do_specconv_time(x, layer_ctr, N, nt, dur_increased_factor)

            skip.append(x)

        T = reversed(self["T"])
        Conv = reversed(self["decode"])

        for t, conv in zip(T, Conv):
            layer_ctr += 1

            x = t(x), skip.pop(-2)
            x = conv(torch.concat(x, 1))

            x = self.do_specconv_time(x, layer_ctr, N, nt, dur_increased_factor)

        x = self["output"](x)

        x = self.unpad(x)

        ########################################################
        _, c, H, W = x.shape
        x = x.reshape(N, nt, c, H, W)  # (batch, nt, 1, nx, nz)

        if self.mean_or_max == "max":
            x = torch.max(x, dim=(1))[0]  # output is (batch, 1, nx, nz)
        elif self.mean_or_max == "mean":
            x = torch.mean(x, dim=(1))  # output is (batch, 1, nx, nz)

        return x


class DnCNN(torch.nn.Module):
    def __init__(self, channels, num_layers=4, features=64):
        super(DnCN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(
            torch.nn.Conv2d(
                in_channels=channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(torch.nn.BatchNorm2d(features))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(
            torch.nn.Conv2d(
                in_channels=features,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class DeepfUS(torch.nn.ModuleDict):
    def __init__(self, params):
        super().__init__()

        class Conv(torch.nn.Sequential):
            def __init__(self, idim, odim):
                super().__init__()

                if params.model_params.residual:
                    self.conv_matching_residual_dim = torch.nn.Conv2d(
                        idim, odim, 1, padding="same"
                    )
                    block_idim = odim
                else:
                    block_idim = idim

                self.block = torch.nn.Sequential(
                    torch.nn.Conv2d(block_idim, odim, 3, padding="same"),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(params.model_params.dropout),
                    torch.nn.Conv2d(odim, odim, 3, padding="same"),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(params.model_params.dropout),
                )

            def forward(self, x):
                if params.model_params.residual:
                    x = self.conv_matching_residual_dim(x)

                y = self.block(x)

                if params.model_params.residual:
                    y += x

                return y

        def Transpose(idim: int, odim: int):
            kwargs = dict(kernel_size=3, stride=2, padding=1, output_padding=1)
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(idim, odim, **kwargs), torch.nn.ReLU()
            )

        self.num_frames = (idim := params.num_frames)

        if params.model_params.conv3d.enable:
            conv3d_kernelsize = int(
                np.ceil(
                    params.model_params.conv3d.kernelsize_relative * self.num_frames
                )
            )
            self["conv"] = torch.nn.Conv3d(
                1,
                out_channels=(params.model_params.conv3d.outch),
                # we have that (T, H, W) to this layer
                kernel_size=(conv3d_kernelsize, 1, 1),
                padding="same",
            )
            idim = params.model_params.conv3d.outch * self.num_frames

        self["input"] = Conv(idim, params.model_params.num_channels[0])
        self["output"] = torch.nn.Conv2d(params.model_params.num_channels[0], 1, 1)

        self["encode"] = torch.nn.ModuleList(
            map(
                Conv,
                params.model_params.num_channels[:-1],
                params.model_params.num_channels[1:],
            )
        )
        self["decode"] = torch.nn.ModuleList(
            map(
                Conv,
                params.model_params.num_channels[1:],
                params.model_params.num_channels[:-1],
            )
        )

        self["T"] = torch.nn.ModuleList(
            map(
                Transpose,
                params.model_params.num_channels[1:],
                params.model_params.num_channels[:-1],
            )
        )

    def do_padding(self, x, divisible=16):
        N, T, H, W = x.shape

        # make sure the H,W is divisible by divisible (2^4) (unet depth)
        h_pad = divisible - H % divisible
        w_pad = divisible - W % divisible
        if h_pad == divisible:
            h_pad = 0
        if w_pad == divisible:
            w_pad = 0

        padding = [w_pad // 2, h_pad // 2]

        # pad both sides
        unpad_list = list()
        for p in padding[::-1]:
            if p == 0:
                unpad_amount_pos = None
                unpad_amount_neg = None
            else:
                unpad_amount_pos = p
                unpad_amount_neg = -p
            unpad_list.append(slice(unpad_amount_pos, unpad_amount_neg, None))
        unpad_indices = (Ellipsis,) + tuple(unpad_list)

        padding = [i for p in padding for i in (p, p)]

        x_padded = F.pad(x, padding, mode="constant")

        out_put_shape = x_padded.shape[2:]

        self._unpad_indices = unpad_indices

        return x_padded

    def unpad(self, x):
        """Remove the padding from padding inputs"""
        unpad_indices = self._unpad_indices
        return x[unpad_indices]

    def forward(self, x):
        N, H, W, T = x.shape

        # (batch, num_frames, h, w)
        x = torch.moveaxis(x, -1, 1)

        x = self.do_padding(x)

        if "conv" in self:
            # for 3d convolution
            # (batch, 1, num_frames, h, w) < -- (batch, num_frames, h, w)
            x = x.unsqueeze(1)
            x = self["conv"](x)
            x = F.relu(x)
            # move back the num_frames into ch dim
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[-2], x.shape[-1])

        skip = [x := self["input"](x)]

        for conv in self["encode"]:
            x = F.max_pool2d(x, 2, 2)
            skip.append(x := conv(x))

        T = reversed(self["T"])
        Conv = reversed(self["decode"])

        for t, conv in zip(T, Conv):
            x = t(x), skip.pop(-2)
            x = conv(torch.concat(x, 1))

        x = self["output"](x)

        x = self.unpad(x)

        # output is (batch, 1, nx, nz)
        assert x.shape == (N, 1, H, W)

        return x
