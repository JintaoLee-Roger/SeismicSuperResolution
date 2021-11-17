import torch.nn as nn
import torch
from model import common

def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = (kernel_size - 1) // 2
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])

    return nn.Sequential(*layers)

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule,self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

def make_model(args):
    return UNet(feature_scale=args.feature_scale, scale=args.scale)

class UNet(nn.Module):
    def __init__(self, n_input_channels=1, n_output_channels=1, feature_scale=1, more_layers=0,
                concat_x=False, upsample_model='deconv', pad='zero', norm_layer=nn.BatchNorm2d,
                need_bias=True, scale=2):

        super(UNet, self).__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x

        features = [64, 128, 256, 512, 1024]
        features = [x // self.feature_scale for x in features]

        self.start = unetConv2d(n_input_channels, 
                                features[0] if not self.concat_x else features[0] - n_input_channels,
                                norm_layer, need_bias, pad)

        self.down1 = unetDown(features[0], features[1] if not self.concat_x else features[1] - n_input_channels,
                              norm_layer, need_bias, pad)
        self.down2 = unetDown(features[1], features[2] if not self.concat_x else features[2] - n_input_channels,
                              norm_layer, need_bias, pad)
        self.down3 = unetDown(features[2], features[3] if not self.concat_x else features[3] - n_input_channels,
                              norm_layer, need_bias, pad)
        self.down4 = unetDown(features[3], features[4] if not self.concat_x else features[4] - n_input_channels,
                              norm_layer, need_bias, pad)

        # more downsampling layers
        if more_layers > 0:
            self.more_downs = [
                     unetDown(features[4], features[4] if not self.concat_x else features[4] - n_input_channels,
                              norm_layer, need_bias, pad) for i in range(self.more_layers)]

            self.more_ups = [
                     unetUp(features[4], upsample_model, need_bias, pad,
                            same_num_feat=True) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)

        self.up4 = unetUp(features[3], upsample_model, need_bias, pad)
        self.up3 = unetUp(features[2], upsample_model, need_bias, pad)
        self.up2 = unetUp(features[1], upsample_model, need_bias, pad)
        self.up1 = unetUp(features[0], upsample_model, need_bias, pad)

        self.upend = common.Upsampler(common.default_conv, scale, features[0], act=False)
        resblock = [common.ResBlock(
            common.default_conv, features[0], 3, act=nn.ReLU(True), bn=True, res_scale=1
            ) for _ in range(3)]
        self.resblock = nn.Sequential(*resblock)

        self.final = conv(features[0], n_output_channels, 1, bias=need_bias, pad=pad)
        
        self.final = nn.Sequential(self.final, nn.Sigmoid())

    def forward(self, inputs):
        
        # DownSample
        downs = [inputs]

        down = nn.AvgPool2d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))

        in64 = self.start(inputs)
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], 1)

        down1 = self.down1(in64)
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], 1)

        down2 = self.down2(down1)
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], 1)

        down3 = self.down3(down2)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], 1)

        down4 = self.down4(down3)
        if self.concat_x:
            down4 = torch.cat([down4, downs[4]], 1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out, downs[kk + 5]], 1)

                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more_layers - idx - 2]
                up_ = l(up_, prevs[self.more_layers - idx - 2])

        else:
            up_ = down4

        up4 = self.up4(up_, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)
        output = self.upend(up1)
        output = self.resblock(output)

        return self.final(output)

class unetConv2d(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2d, self).__init__()

        if norm_layer is not None:
            self.conv1 = nn.Sequential(
                    conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                    norm_layer(out_size),
                    nn.ReLU()
                    )
            self.conv2 = nn.Sequential(
                    conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                    norm_layer(out_size),
                    nn.ReLU()
                    )
        else:
            self.conv1 = nn.Sequential(
                    conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                    nn.ReLU()
                    )
            self.conv2 = nn.Sequential(
                    conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                    nn.ReLU()
                    )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetDown, self).__init__()

        self.conv = unetConv2d(in_size, out_size, norm_layer, need_bias, pad)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)

        return outputs

class unetUp(nn.Module):
    def __init__(self, out_size, upsample_model, need_bias, pad, same_num_feat=False):
        super(unetUp, self).__init__()

        n_feat = out_size if same_num_feat else out_size * 2
        if upsample_model == 'deconv':
            self.up = nn.ConvTranspose2d(n_feat, out_size, 4, stride=2, padding=1)
            self.conv = unetConv2d(out_size * 2, out_size, None, need_bias, pad)
        elif upsample_model == 'bilinear' or upsample_model == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_model),
                    conv(n_feat, out_size, 3, bias=need_bias, pad=pad))
            self.conv = unetConv2d(out_size * 2, out_size, None, need_bias, pad)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)

        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output = self.conv(torch.cat([in1_up, inputs2_], 1))

        return output





