import tensorflow as tf

from keras.layers import *
from keras.models import *

from layers.residual_block import ResidualBlock


_3d_base_params = {
    'point_size': 3,
    'downward_convs': [1, 1, 2, 3, 5],
    'downward_filters': [4, 16, 64, 128, 256],
    'upward_convs': [4, 3, 2, 1],
    'upward_filters': [128, 64, 16, 4],
    'resolution': 64,
}


class Agg3D(Model):
    """
    The complete mu-Net model
    """
    def __init__(
            self,
            downward_convs=None,
            downward_filters=None,
            upward_convs=None,
            upward_filters=None,
            resolution=None,
            *args, **kwargs
    ):
        """
        Initialises mu-Net
        Args:
            point_size: The size of the cube being placed within the voxel grid
            downward_convs: The number of convolutional blocks at each downard stage of the U-Net
            downward_filters: The number of filters for each of the downward stages
            upward_convs: The number of convolutional blocks at each upward stage of the U-Net
            upward_filters: The number of filters for each of the upward stages
            resolution: The resolution of the voxel grid outputted by mu-Net
            threshold: A threshold for when to consider a muon to not scatter
            poca_nn: Use an (already trained) neural network to compute the scattering points?
            hidden_layers: The number of hidden layers of the MLP used to process the muon detections (as a multiple of
            the projection dim)
            use_lstm: Use an LSTM to process the entire set of muon detections together?
            use_residual: Use Residual Blocks? (default is ConvNeXt)
            non_scattered_mul: At which point along the trajectory of the non-scattered muons should they be placed
            (as a fraction of total distance travelled), put None for random placement
        """

        super(Agg3D, self).__init__(*args, **kwargs)
        self.resolution = resolution
        self.downward_filters = downward_filters

        conv = ResidualBlock

        # downward ConvNeXt blocks
        self.downward_convs = []
        for stage in range(len(downward_convs)):
            stack = []
            for c in range(downward_convs[stage]):
                stack.append(conv(
                    downward_filters[stage],
                    kernel_size=3,
                    dims=2,
                    name=f'{self.name}-downward_stage_{stage}-block_{c}'
                ))
            self.downward_convs.append(Sequential(stack, name=f'{self.name}-downward_stage_{stage}'))

        self.downsampling = []
        self.upsampling = []
        self.upward_convs = []
        for stage in range(len(downward_convs) - 1):
            # downsampling convolutions
            self.downsampling.append(Sequential([
                LayerNormalization(epsilon=1e-6, name=f'{self.name}-downward_stage_{stage}-downsampling-layer_norm'),
                Conv2D(filters=downward_filters[stage + 1],
                       kernel_size=2,
                       strides=2,
                       name=f'{self.name}-downward_stage_{stage}-downsampling-conv2d',
                       padding='same')
            ], name=f'{self.name}-downward_stage_{stage}-downsampling'))

            # upsampling + convolutions
            self.upsampling.append(Sequential([
                Dense(units=upward_filters[stage], name=f'{self.name}-upward_stage_{stage}-upsampling-pointwise_conv'),
                LayerNormalization(epsilon=1e-6, name=f'{self.name}-upward_stage_{stage}-upsampling-layer_norm'),
                UpSampling2D(name=f'{self.name}-upward_stage_{stage}-upsampling-upsampling'),
            ]))

            # upward ConvNeXt blocks
            stack = []
            for c in range(upward_convs[stage]):
                stack.append(conv(
                    upward_filters[stage],
                    kernel_size=3,
                    dims=2,
                    name=f'{self.name}-upward_stage_{stage}-block_{c}'
                ))
            self.upward_convs.append(Sequential(stack, name=f'{self.name}-upward_stage_{stage}'))

        self.final_conv = Conv2D(1, 1, name=f'{self.name}-final_conv')

    def call(self, inputs, training=None, mask=None):
        # data format of inputs is theta_edges, phi_edges, transmission
        x = inputs

        skip_outputs = []
        for i, block in enumerate(self.downward_convs):
            x = block(x)
            if i != len(self.downsampling):
                skip_outputs.append(x)
                x = self.downsampling[i](x)

        for i, block in enumerate(self.upward_convs):
            x = self.upsampling[i](x)
            x += skip_outputs.pop()
            x = block(x)

        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    model = Agg3D(
        **{
            'downward_convs': [1, 2, 3, 4, 5],
            'downward_filters': [2*8, 2*16, 2*32, 2*64, 2*128],
            'upward_convs': [4, 3, 2, 1],
            'upward_filters': [2*64, 2*32, 2*16, 2*8],
            'resolution': 64,
        }
    )
    print(model(tf.random.normal((8, 20000, 13))).shape)

    model.summary()