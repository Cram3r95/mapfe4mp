import torch
from torch import nn

from model.utils.load_models import load_VGG_model
from model.modules.encoders import EncoderLSTM as Encoder

class VisualExtractor(nn.Module):
    """
    In this class add all the visual extractors that will be used in the architecture.
    Currently supporting:
        - VGG{11, 13, 16, 19}
    """
    def __init__(self, visual_extractor_type, config=None):
        super(VisualExtractor, self).__init__()
        if visual_extractor_type == "vgg19":
            self.module = VGG(config)
        elif visual_extractor_type == "home":
            self.module = HOME()
        else:
            raise NotImplementedError("Unknown visual extractor module {}.".format(visual_extractor_type))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class VGG(nn.Module):

    def __init__(self, config):
        super(VGG, self).__init__()
        self.module = load_VGG_model(**config)

    def forward(self, data_input):
        """
            Input shape: Batch, Channels_in, H_in, W_in
            Output shape: Batch, Channels_out, H_out, W_out
            ?> CHECK DIMENSION BEFORE FORWARD
        """
        image_feature_map = self.module(data_input)
        return image_feature_map

class HOME(nn.Module):

    def __init__(self, ch_list=[3,32,64,128,128]):
        super().__init__()
        assert len(ch_list) == 5, "ch_list must have 5 elements"
        self.m = nn.Sequential(
            nn.Conv2d(ch_list[0],ch_list[1],3,1,1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(ch_list[1],ch_list[2],3,1,1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(ch_list[2],ch_list[3],3,1,1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(ch_list[3],ch_list[4],3,1,1),
            nn.ReLU()
        )

    def forward(self, x):
        """
            x: (b,3,224,224)
            return (b,128,28,28)
        """
        return self.m(x)

class CNN(nn.Module):
    def __init__(self,
                 social_pooling=False,
                 channels_cnn=4,
                 mlp=32,
                 encoder_h_dim=16,
                 insert_trajectory=False,
                 need_decoder=False,
                 PhysFeature=False,
                 grid_size_in=32,
                 grid_size_out=32,
                 num_layers=3,
                 dropout=0.,
                 batch_norm=False,
                 non_lin_cnn="tanh",
                 in_channels=3,
                 skip_connection=False,
                 ):
        super(CNN, self).__init__()
        self.__dict__.update(locals())

        self.bottleneck_dim = int(grid_size_in / 2 ** (num_layers - 1)) ** 2

        num_layers_dec = int(num_layers + ((grid_size_out - grid_size_in) / grid_size_out))

        # Encoder
        
        self.encoder = nn.Sequential()

        layer_out = channels_cnn
        self.encoder.add_module("ConvBlock_1", Conv_Blocks(in_channels, channels_cnn,
                                                           dropout=dropout,
                                                           batch_norm=batch_norm,
                                                           non_lin=self.non_lin_cnn,
                                                           first_block=True,
                                                           skip_connection=self.skip_connection

                                                           ))
        layer_in = layer_out
        for layer in np.arange(2, num_layers + 1):

            if layer != num_layers:
                layer_out = layer_in * 2
                last_block = False
            else:
                layer_out = layer_in
                last_block = True
            self.encoder.add_module("ConvBlock_%s" % layer,
                                    Conv_Blocks(layer_in, layer_out,
                                                dropout=dropout,
                                                batch_norm=batch_norm,
                                                non_lin=self.non_lin_cnn,
                                                skip_connection=self.skip_connection,
                                                last_block=last_block
                                                ))
            layer_in = layer_out

        self.bootleneck_channel = layer_out
        if self.need_decoder:

            self.decoder = nn.Sequential()
            layer_in = layer_out
            for layer in range(1, num_layers_dec + 1):
                first_block = False
                extra_d = 0
                layer_in = layer_out
                last_block = False
                filter = 4
                padding = 1
                if layer == 1:
                    if self.insert_trajectory:
                        extra_d = 1

                    first_block = True
                    layer_out = layer_in

                else:
                    layer_out = int(layer_in / 2.)

                if layer == num_layers_dec:
                    layer_out = 1
                    last_block = True
                    padding = 0
                    filter = 3

                self.decoder.add_module("UpConv_%s" % layer,
                                        UpConv_Blocks(int(layer_in + extra_d),
                                                      layer_out,
                                                      first_block=first_block,
                                                      filter=filter,
                                                      padding=padding,
                                                      dropout=dropout,
                                                      batch_norm=batch_norm,
                                                      non_lin=self.non_lin_cnn,
                                                      skip_connection=self.skip_connection,
                                                      last_block=last_block))

        if self.insert_trajectory:
            self.traj2cnn = make_mlp(
                dim_list=[encoder_h_dim, mlp, self.bottleneck_dim],
                activation_list=["tanh", "tanh"],
            )

        self.init_weights()

    def init_weights(self):
        def init_kaiming(m):
            if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                m.bias.data.fill_(0.01)
            # if type(m) in [nn.ConvTranspose2d]:
            # torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
            # m.bias.data.fill_(50)

        def init_xavier(m):
            if type(m) == [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        if self.non_lin_cnn in ['relu', 'leakyrelu']:
            self.apply(init_kaiming)
        elif self.non_lin_cnn == "tanh":
            self.apply(init_xavier)
        else:
            assert False, "non_lin not valid for initialisation"

    def forward(self, image, traj_h=torch.empty(1), pool_h=torch.empty(1)):
        output = {}

        enc = self.encoder(image)

        if self.PhysFeature:
            # enc_out = self.leakyrelu(self.encoder_out(enc))
            # enc_out = enc_out.permute(1, 0, 2, 3).view(1, enc_out.size(0), -1)
            output.update(Features=enc)

        if self.need_decoder:

            if self.skip_connection:
                batch, c, w, h = enc[0].size()
                in_decoder, skip_con_list = enc

            else:
                batch, c, w, h = enc.size()
                in_decoder = enc

            if self.insert_trajectory:

                traj_enc = self.traj2cnn(traj_h)

                traj_enc = traj_enc.view(batch, 1, w, h)
                in_decoder = torch.cat((traj_enc, in_decoder), 1)
            if self.social_pooling:

                social_enc = self.social_states(pool_h)

                social_enc = social_enc.view(batch, 1, w, h)
                in_decoder = torch.cat((social_enc, in_decoder), 1)
            if self.skip_connection: in_decoder = [in_decoder, skip_con_list]
            dec = self.decoder(in_decoder)
            output.update(PosMap=dec)

        return output