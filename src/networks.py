"""
CycleGAN Network Architectures & Utilities (PyTorch + MONAI).

Implements:
1. `resnet_6blocks`: 6-block ResNet Generator (default CycleGAN settings from junyanz/pytorch-CycleGAN-and-pix2pix).
2. `unet_128`: 7-downsample U-Net Generator (128x128 input -> 1x1 bottleneck with skip connections).
3. `monai_resunet`: MONAI native Residual U-Net Generator (`monai.networks.nets.UNet` with residual units + InstanceNorm).
4. `NLayerDiscriminator`: 70x70 PatchGAN Discriminator (`n_layers=3`, `ndf=64`).
5. `GANLoss` (LSGAN MSE objective) and `ImagePool` (50-image history buffer).
"""

from __future__ import annotations

import functools
import random
from typing import List, Optional, Type

import torch
import torch.nn as nn
from torch.nn import init

from monai.networks.layers import Norm
from monai.networks.nets import UNet as MONAIUNet


def get_norm_layer(norm_type: str = "instance") -> Type[nn.Module]:
    """Return normalization layer constructor matching CycleGAN defaults."""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError(f"Normalization layer [{norm_type}] is not found")


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialize network weights using normal N(0, 0.02) as in standard CycleGAN."""

    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class ResnetBlock(nn.Module):
    """Residual block with ReflectionPad2d and InstanceNorm2d."""

    def __init__(
        self,
        dim: int,
        norm_layer: Type[nn.Module],
        use_dropout: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        conv_block: List[nn.Module] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))
        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """
    ResNet-based Generator (`resnet_6blocks` or `resnet_9blocks`).
    Consists of 2 downsampling layers, N residual blocks at 1/4 spatial resolution,
    and 2 upsampling layers with ReflectionPad2d and Tanh output activation.
    """

    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        ngf: int = 64,
        norm_layer: Type[nn.Module] = nn.InstanceNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 6,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # 2 Downsampling layers (128x128 -> 64x64 -> 32x32)
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        # N Residual blocks (n_blocks = 6 for resnet_6blocks)
        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model.append(
                ResnetBlock(
                    ngf * mult,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            )

        # 2 Upsampling layers (32x32 -> 64x64 -> 128x128)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    """U-Net submodule with skip connection (used recursively by `UnetGenerator`)."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: Optional[int] = None,
        submodule: Optional[nn.Module] = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: Type[nn.Module] = nn.InstanceNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """
    U-Net Generator (`unet_128` with `num_downs=7` for 128x128 input).
    Constructed recursively from the 1x1 innermost bottleneck outward.
    """

    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        num_downs: int = 7,
        ngf: int = 64,
        norm_layer: Type[nn.Module] = nn.InstanceNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MONAIResUNetGenerator(nn.Module):
    """
    MONAI-native Residual U-Net Generator (`monai.networks.nets.UNet`).
    Combines multi-scale U-Net skip connections (4 resolution stages: 128->64->32->16)
    with 2 residual units per stage and Instance Normalization, avoiding the severe
    1x1 bottleneck compression of vanilla `unet_128`.
    """

    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        ngf: int = 64,
        num_res_units: int = 2,
    ) -> None:
        super().__init__()
        self.unet = MONAIUNet(
            spatial_dims=2,
            in_channels=input_nc,
            out_channels=output_nc,
            channels=(ngf, ngf * 2, ngf * 4, ngf * 8),
            strides=(2, 2, 2),
            num_res_units=num_res_units,
            norm=Norm.INSTANCE,
            act=("LEAKYRELU", {"negative_slope": 0.2, "inplace": True}),
        )
        self.out_act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.unet(x))


class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN Discriminator matching standard CycleGAN."""

    def __init__(
        self,
        input_nc: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: Type[nn.Module] = nn.InstanceNorm2d,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence: List[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def define_G(
    netG: str,
    input_nc: int = 1,
    output_nc: int = 1,
    ngf: int = 64,
    norm: str = "instance",
    use_dropout: bool = False,
    init_type: str = "normal",
    init_gain: float = 0.02,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Factory function for CycleGAN generators."""
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == "resnet_6blocks":
        net: nn.Module = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6
        )
    elif netG == "resnet_9blocks":
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9
        )
    elif netG == "unet_128":
        net = UnetGenerator(
            input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
    elif netG == "monai_resunet":
        net = MONAIResUNetGenerator(input_nc, output_nc, ngf=ngf, num_res_units=2)
    else:
        raise ValueError(f"Generator architecture [{netG}] is not recognized.")

    net = net.to(device)
    init_weights(net, init_type=init_type, init_gain=init_gain)
    return net


def define_D(
    input_nc: int = 1,
    ndf: int = 64,
    n_layers_D: int = 3,
    norm: str = "instance",
    init_type: str = "normal",
    init_gain: float = 0.02,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Factory function for 70x70 PatchGAN discriminator."""
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer).to(device)
    init_weights(net, init_type=init_type, init_gain=init_gain)
    return net


class GANLoss(nn.Module):
    """GAN Loss supporting LSGAN (MSE, default in CycleGAN) and Vanilla GAN (BCE)."""

    def __init__(self, gan_mode: str = "lsgan", target_real_label: float = 1.0, target_fake_label: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"gan mode {gan_mode} not implemented")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class ImagePool:
    """
    Image buffer that stores previously generated images to update discriminators
    using a history of 50 generated images (Shrivastava et al., 2017; Zhu et al., 2017).
    """

    def __init__(self, pool_size: int = 50) -> None:
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images: List[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        return_images: List[torch.Tensor] = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)
