import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.wavenet.func import dcnn_build, wavenet_build, dcnn_build2
import tensorflow as tf
from tensorflow import keras
#%%

if __name__ == '__main__':
    cfg = {"n_steps_in": 50, "n_steps_out": 3, 'n_layers': 4, "n_filters": 4,
           "n_kernel": 3, "n_epochs": 100, "n_batch": 100, 'hidden_channels': 4, 'reg': None}
    keras_model = dcnn_build(cfg, 1)
    keras_model.summary()
    tf.keras.utils.plot_model(
        keras_model, to_file='model_dcnn.png', show_shapes=False, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )

    keras_model = dcnn_build2(cfg, 3)
    keras_model.summary()
    tf.keras.utils.plot_model(
        keras_model, to_file='model_dcnn2.png', show_shapes=True, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )


    keras_model = wavenet_build(cfg, 1)
    keras_model.summary()
    tf.keras.utils.plot_model(
        keras_model, to_file='model_wavenet.png', show_shapes=False, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )

    # keras_model.compile(loss=keras.losses.mean_squared_error,
    #                     # optimizer=keras.optimizers.RMSprop(),  # lr=2e-5
    #                     optimizer=keras.optimizers.Adam(),  # lr=2e-5
    #                     metrics=['mse'])

    # model_dir = os.path.join(os.getcwd(), 'models', 'dcnn_seq')
    # os.makedirs(model_dir, exist_ok=True)
    # print("model_dir: ", model_dir)
    # estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir=model_dir)