# from doom_rl.models.model import DqnKerasModel
#
# import keras
# from tensorflow.keras.initializers import Constant, glorot_normal
# from tensorflow.keras.layers import Conv2D, Dense, Flatten
#
#
# class SimpleKerasModel(DqnKerasModel):
#     def __init__(self, state_shape, nb_actions, process_state_batch):
#         super(SimpleKerasModel, self).__init__(state_shape, nb_actions, process_state_batch=process_state_batch)
#
#     def _build_network(self):
#         conv1 = Conv2D(24, 6, strides=(3, 3), activation="relu",
#                        kernel_initializer=glorot_normal(),
#                        bias_initializer=Constant(0.01), name="ConvLayer1")(self.input_layer)
#         conv2 = Conv2D(32, 3, strides=(2, 2), activation="relu",
#                        kernel_initializer=glorot_normal(),
#                        bias_initializer=Constant(0.01), name="ConvLayer2")(conv1)
#         conv_flat = Flatten()(conv2)
#         fc1 = Dense(128, activation="relu",
#                     kernel_initializer=glorot_normal(),
#                     bias_initializer=Constant(0.01), name="FullyConnected1")(conv_flat)
#         self.output_layer = Dense(self.nb_actions, activation="relu",
#                                   kernel_initializer=glorot_normal(),
#                                   bias_initializer=Constant(0.01), name="FullyConnected2")(fc1)
#
#
# # A toy model which requires less computation
# class SimplerKerasModel(DqnKerasModel):
#     def __init__(self, state_shape, nb_actions, process_state_batch):
#         super(SimplerKerasModel, self).__init__(state_shape, nb_actions, process_state_batch=process_state_batch)
#
#     def _build_network(self):
#         conv1 = Conv2D(6, 6, strides=(3, 3), activation="relu",
#                        kernel_initializer=glorot_normal(),
#                        bias_initializer=Constant(0.01), name="ConvLayer1")(self.input_layer)
#         conv2 = Conv2D(12, 3, strides=(2, 2), activation="relu",
#                        kernel_initializer=glorot_normal(),
#                        bias_initializer=Constant(0.01), name="ConvLayer2")(conv1)
#         conv_flat = Flatten()(conv2)
#         self.output_layer = Dense(self.nb_actions, activation="relu",
#                                   kernel_initializer=glorot_normal(),
#                                   bias_initializer=Constant(0.01), name="FullyConnected1")(conv_flat)
