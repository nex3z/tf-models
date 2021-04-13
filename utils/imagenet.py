from utils.imagenet_labels import LABELS


def validate_input_shape(weights, input_shape, default_size):
    if weights == 'imagenet' and input_shape != default_size:
        raise ValueError(f"When loading imagenet weights, input_shape should be {default_size}")


def decode_prediction(prediction, top=1):
    top_idxes = prediction.argsort()[-top:][::-1]
    return tuple([(idx, LABELS[idx], prediction[idx]) for idx in top_idxes])
