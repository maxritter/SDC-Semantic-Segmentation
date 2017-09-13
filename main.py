# Import section
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Load pretrained VGG Model into TensorFlow
def load_vgg(sess, vgg_path):
    # Specify names for the layers
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load VGG model from saved file
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Extract the layers we need to create our new network
    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input, keep, layer3, layer4, layer7

# Create the layers for a fully convolutional network and build skip-layers using the vgg layers
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    # Define our kernel initializer and regularizer
    init = tf.truncated_normal_initializer(stddev = 0.01)
    reg = tf.contrib.layers.l2_regularizer(1e-3)

    # Do 1x1 convolutions on layer 3, 4 and 7 with L2 regularizer for the weights
    conv_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                   kernel_initializer=init, kernel_regularizer=reg)
    conv_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                   kernel_initializer=init, kernel_regularizer=reg)
    conv_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                   kernel_initializer=init, kernel_regularizer=reg)

    # Do our first transposed convolution from layer 7
    deconv_1 = tf.layers.conv2d_transpose(conv_layer7, num_classes, 4, 2, padding='same',
                                          kernel_initializer=init, kernel_regularizer=reg)

    # Add the first skip connection from layer 4
    skip_1 = tf.add(deconv_1, conv_layer4)

    # Do our second transposed convolution on that result
    deconv_2 = tf.layers.conv2d_transpose(skip_1, num_classes, 4, 2, padding='same',
                                          kernel_initializer=init, kernel_regularizer=reg)

    # Add the second skip connection from layer 3
    skip_2 = tf.add(deconv_2, conv_layer3)

    # Do our third and last transposed convolution to match input image size
    deconv_3 = tf.layers.conv2d_transpose(skip_2, num_classes, 16, 8, padding='same',
                                          kernel_initializer=init, kernel_regularizer=reg)
    return deconv_3

# Build the TensorFLow loss and optimizer operations
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    # Logits is a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # Those are our output labels, reshaped to match size
    labels = tf.reshape(correct_label, (-1, num_classes))

    # We use standard cross-entropy-loss as our loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # For the optimizer, we use Adam as it is a good general choice
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

# Train neural network and print out loss
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

    # Init our global variables
    sess.run(tf.global_variables_initializer())

    # Go through all epochs
    for epoch in range(epochs):
        # Print out epoch
        print("Epoch {}".format(epoch + 1), "/ {} ..".format(epochs))

        # Go through all batches
        batch = 1
        for image, label in get_batches_fn(batch_size):
            # Print out batch number and raise it
            print("Batch {} ..".format(batch))
            batch = batch + 1

            # Train our model and get loss
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label,
                                          keep_prob: 0.8, learning_rate: 1e-4})

        # Print loss for each epoch
        print("Epoch {}".format(epoch + 1), " loss: {:.4f}".format(loss))

# Train our FCN for road semantic segmentation
def run():
    # Configuration
    num_classes = 2
    image_shape = (160, 576)
    data_dir = os.path.abspath('data')
    runs_dir = os.path.abspath('runs')

    # Eventually download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Hyper-Parameter
    batch_size = 1
    epochs = 10
    learning_rate = tf.constant(1e-4)

    # Run tensorflow session
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Load pretrained VGG Model into TensorFlow and extract layers
        print("Loading VGG model as encoder..")
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # Create our FCN model
        print("Creating our decoder part on top..")
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # Build the TensorFlow loss and optimizer operations
        print("Create loss and optimizer..")
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # Train our model using the train_nn function
        print("Train our network..")
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        print("Save inference samples..")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

# This is our main entry point
if __name__ == '__main__':
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion(
        '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    # Print start screen
    print("*** Semantic Segmentation by Max Ritter ***")

    # Eventually run some tests
    tests.test_for_kitti_dataset(os.path.abspath('data'))
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)

    # Train our network and do the inference
    run()