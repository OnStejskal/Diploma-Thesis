import torch
import torch.nn as nn
import torch.nn.functional as F

# He initializer for the layers with ReLU activation function:
def he_init(tensor):
    nn.init.kaiming_normal_(tensor, nonlinearity='relu')

b_init = torch.zeros


class SAUNet(nn.Module):
    def __init__(self, n_out, is_training, n_filters=64, n_out_att=None, use_bn=True, upsampling_mode='NN',
                 name='SelfAttentionUNet'):
        super(SAUNet, self).__init__()

        assert upsampling_mode in ['deconv', 'NN']

        # network parameters
        self.n_out = n_out
        self.is_training = is_training
        self.nf = n_filters
        self.use_bn = use_bn
        self.upsampling_mode = upsampling_mode
        self.name = name

        if n_out_att is None:
            self.n_out_att = n_out
        else:
            self.n_out_att = n_out_att

        # final prediction
        self.prediction = None

        # multi-scale (multi-channel) attention maps:
        self.attention_tensors = None

    def forward(self, incoming):
        encoder = self.build_encoder(incoming)
        code = self.build_bottleneck(encoder)
        decoder = self.build_decoder(code)
        self.prediction = self.build_output(decoder)

        return self.prediction

    def build_encoder(self, incoming):
        """ Encoder layers """

        # check for compatible input dimensions
        shape = incoming.size()
        assert shape[2] % 16 == 0
        assert shape[3] % 16 == 0

        encoder_layers = []
        en_brick_0, concat_0 = self._encode_brick(incoming, self.nf, self.is_training,
                                                    scope='encode_brick_0', use_bn=self.use_bn)
        encoder_layers.extend([en_brick_0, concat_0])

        en_brick_1, concat_1 = self._encode_brick(en_brick_0, 2 * self.nf, self.is_training,
                                                    scope='encode_brick_1', use_bn=self.use_bn)
        encoder_layers.extend([en_brick_1, concat_1])

        en_brick_2, concat_2 = self._encode_brick(en_brick_1, 4 * self.nf, self.is_training,
                                                    scope='encode_brick_2', use_bn=self.use_bn)
        encoder_layers.extend([en_brick_2, concat_2])

        en_brick_3, concat_3 = self._encode_brick(en_brick_2, 8 * self.nf, self.is_training,
                                                    scope='encode_brick_3', use_bn=self.use_bn)
        encoder_layers.extend([en_brick_3, concat_3])

        return encoder_layers

    def build_bottleneck(self, encoder):
        """ Central layers """
        en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3 = encoder

        code = self._bottleneck_brick(en_brick_3, 16 * self.nf, self.is_training, scope='code', use_bn=self.use_bn)

        return [en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3, code]

    def build_decoder(self, code):
        """ Decoder layers """
        en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3, code = code

        decoder_layers = []
        dec_brick_0, attention_0 = self._decode_brick(code, concat_3, 8 * self.nf, self.is_training,
                                                        scope='decode_brick_0', use_bn=self.use_bn)
        decoder_layers.extend([dec_brick_0, attention_0])

        dec_brick_1, attention_1 = self._decode_brick(dec_brick_0, concat_2, 4 * self.nf, self.is_training,
                                                        scope='decode_brick_1', use_bn=self.use_bn)
        decoder_layers.extend([dec_brick_1, attention_1])

        dec_brick_2, attention_2 = self._decode_brick(dec_brick_1, concat_1, 2 * self.nf, self.is_training,
                                                        scope='decode_brick_2', use_bn=self.use_bn)
        decoder_layers.extend([dec_brick_2, attention_2])

        dec_brick_3, attention_3 = self._decode_brick(dec_brick_2, concat_0, self.nf, self.is_training,
                                                        scope='decode_brick_3', use_bn=self.use_bn)
        decoder_layers.extend([dec_brick_3, attention_3])

        self.attention_tensors = [attention_2, attention_1, attention_0]

        return decoder_layers

    def build_output(self, decoder):
        """ Output layers """
        dec_brick_3 = decoder[-2]  # Output is based on the last decoded layer

        # output linear
        return self._output_layer(dec_brick_3, n_channels_out=self.n_out, scope='output')

    def get_prediction(self, one_hot=False, softmax=False):
        if one_hot:
            _, predicted_classes = torch.max(self.prediction, dim=1)
            return F.one_hot(predicted_classes, num_classes=self.n_out)
        if softmax:
            return F.softmax(self.prediction, dim=1)
        return self.prediction




    def _encode_brick(self, incoming, nb_filters, use_bn=True):
        """ Encoding brick: conv --> conv --> max pool.
        """
        conv1 = nn.Conv2d(incoming.size(1), nb_filters, kernel_size=3, stride=1, padding=1)
        conv1.weight.data = he_init(conv1.weight.data)
        conv1.bias.data = b_init(conv1.bias.data)

        if use_bn:
            conv1 = nn.BatchNorm2d(nb_filters)
            conv1_act = F.relu(conv1(incoming))
        else:
            conv1_act = F.relu(conv1(incoming))

        conv2 = nn.Conv2d(conv1_act.size(1), nb_filters, kernel_size=3, stride=1, padding=1)
        conv2.weight.data = he_init(conv2.weight.data)
        conv2.bias.data = b_init(conv2.bias.data)

        if use_bn:
            conv2 = nn.BatchNorm2d(nb_filters)
            conv2_act = F.relu(conv2(conv1_act))
        else:
            conv2_act = F.relu(conv2(conv1_act))

        pool = F.max_pool2d(conv2_act, kernel_size=2, stride=2, padding=0)

        concat_layer_out = conv2_act

        return pool, concat_layer_out
        



    
    def _decode_brick(self, incoming, concat_layer_in, nb_filters, is_training, scope, use_bn=True):
        """ Decoding brick: deconv (up-pool) --> conv --> conv.
        """
        with torch.no_grad():  # Needed for correct behavior during inference
            if self.upsampling_mode == 'deconv':
                upsampled = nn.ConvTranspose2d(incoming.size(1), nb_filters, kernel_size=2, stride=2, padding=0)
                upsampled.weight.data = he_init(upsampled.weight.data)
                upsampled.bias.data = b_init(upsampled.bias.data)
                conv1 = upsampled
            else:
                old_height, old_width = incoming.size(2), incoming.size(3)
                new_height, new_width = int(2.0 * old_height), int(2.0 * old_width)
                upsampled = F.interpolate(incoming, size=(new_height, new_width), mode='nearest')

                conv1 = nn.Conv2d(incoming.size(1), nb_filters, kernel_size=3, stride=1, padding=1)
                conv1.weight.data = he_init(conv1.weight.data)
                conv1.bias.data = b_init(conv1.bias.data)

            if use_bn:
                conv1 = nn.BatchNorm2d(nb_filters)
                conv1_act = F.relu(conv1(upsampled))
            else:
                conv1_act = conv1(upsampled)  # first without activation

            concat = torch.cat([conv1_act, concat_layer_in], dim=1)

            conv2 = nn.Conv2d(concat.size(1), nb_filters, kernel_size=3, stride=1, padding=1)
            conv2.weight.data = he_init(conv2.weight.data)
            conv2.bias.data = b_init(conv2.bias.data)
            if use_bn:
                conv2 = nn.BatchNorm2d(nb_filters)
                conv2_act = F.relu(conv2(concat))
            else:
                conv2_act = F.relu(conv2(concat))

            conv3 = nn.Conv2d(conv2_act.size(1), nb_filters, kernel_size=3, stride=1, padding=1)
            conv3.weight.data = he_init(conv3.weight.data)
            conv3.bias.data = b_init(conv3.bias.data)
            if use_bn:
                conv3 = nn.BatchNorm2d(nb_filters)
                conv3_act = F.relu(conv3(conv2_act))
            else:
                conv3_act = F.relu(conv3(conv2_act))

            attention = nn.Conv2d(conv2_act.size(1), self.n_out_att, kernel_size=1, stride=1, padding=0)
            attention.weight.data = he_init(attention.weight.data)
            attention.bias.data = b_init(attention.bias.data)
            attention = F.softmax(attention(conv2_act), dim=1)
            attention_map = torch.sum(attention[..., 1:], dim=1, keepdim=True)

            conv_and_att = conv3_act * attention_map

        return conv_and_att, attention
