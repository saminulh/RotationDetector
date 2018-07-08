import tensorflow as tf
import numpy as np
import os

folder = "Data/"+os.path.basename(__file__).split(".")[0]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

pic_index = 4


def mess_with_data(x, y):
    x = x[y.flatten() == pic_index]
    y = y[y.flatten() == pic_index]

    y *= 0
    temp = x
    s = y.shape[0]
    for i in range(1):
        temp = np.rot90(temp, 1, (1, 2))
        x = np.concatenate((x, temp), axis=0)
        y = np.concatenate((y, np.zeros((s, 1))+i+1), axis=0)

    y = y.astype(int)

    return x, y


x_train, y_train = mess_with_data(x_train, y_train)

x_test, y_test = mess_with_data(x_test, y_test)

my_fc = [tf.feature_column.numeric_column(key="w/e", shape=(32, 32, 3))]


def my_model(features, labels, mode, params):
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.

    net = tf.feature_column.input_layer(features, params['feature_columns'])
    net = tf.reshape(net, (-1, 32, 32, 3))

    for filters, kernel_size, pool_size in params['conv_layers']:
        #net = tf.concat([net, tf.image.rot90(net, 2)], axis=3)
        net = tf.layers.conv2d(inputs=net, filters=filters, kernel_size=kernel_size, activation=tf.nn.relu,
                               padding="same")
        net = tf.layers.max_pooling2d(
            inputs=net, pool_size=pool_size, strides=pool_size)

    net = tf.layers.flatten(net)
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, 20, activation=tf.nn.relu)
    logits = tf.layers.dense(net, 2, activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits, axis=1),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def f():
    temp = tf.data.Dataset.from_tensor_slices(
        ({"w/e": x_train}, y_train)).shuffle(10000000).repeat().batch(100)
    # print(temp)
    return temp


estimator = tf.estimator.Estimator(
    model_fn=my_model,
    params={
        'feature_columns': my_fc,
        'conv_layers': [(64, 5, 2), (128, 5, 2)],
    }, model_dir=folder)


if True:
    estimator.train(
        input_fn=lambda: f(),
        steps=10000)

a = [i["class_ids"] for i in
     estimator.predict(
    input_fn=lambda: tf.data.Dataset.from_tensor_slices(
        ({"w/e": x_test}, y_test)
    ).batch(100)
)
]

print(len(a))
print(x_test.shape)
print("we good")

print(estimator.evaluate(input_fn=lambda: tf.data.Dataset.from_tensor_slices(
    ({"w/e": x_train}, y_train)).batch(100)))
print(estimator.evaluate(input_fn=lambda: tf.data.Dataset.from_tensor_slices(
    ({"w/e": x_test}, y_test)).batch(100)))
