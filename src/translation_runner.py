from src.translation_data_processor import DataProcessor
from src.seq2seq_model import EncoderDecoderModel
import tensorflow as tf


def train(data_path, batch_size=64, n_epoch=200):
    """
    load data, train model
    :param data_path:
    :param batch_size:
    :param n_epoch:
    :return: model
    """
    # load data
    dp = DataProcessor()
    input_ctoi_dict, target_ctoi_dict, input_itoc_dict, target_itoc_dict, encoder_input, decoder_input, decoder_output = dp.load_training_data(data_path)
    samples = encoder_input.shape[0]
    out_steps = decoder_output.shape[1]
    out_dims = decoder_output.shape[2]

    # define dataset
    encoder_input_placeholder = tf.placeholder(encoder_input.dtype, encoder_input.shape)
    decoder_input_placeholder = tf.placeholder(decoder_input.dtype, decoder_input.shape)
    decoder_output_placeholder = tf.placeholder(decoder_output.dtype, decoder_output.shape)
    dataset = tf.data.Dataset.from_tensor_slices((encoder_input_placeholder, decoder_input_placeholder, decoder_output_placeholder))
    dataset = dataset.batch(batch_size).repeat(n_epoch)
    data_iterator = dataset.make_initializable_iterator()
    # for one mini-batch step
    encoder_input_tensor, decoder_input_tensor, decoder_output_tensor = data_iterator.get_next()

    # define training step
    model = EncoderDecoderModel(out_steps, out_dims)
    model_output, loss = model(encoder_input_tensor, decoder_input_tensor, decoder_output_tensor)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    import math
    n_batch = int(math.ceil(samples * 1. / batch_size))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # init data iterator
        sess.run(data_iterator.initializer, feed_dict={encoder_input_placeholder: encoder_input,
                                                       decoder_input_placeholder: decoder_input,
                                                       decoder_output_placeholder: decoder_output})
        for epoch in range(n_epoch):
            for batch in range(n_batch):
                l, _ = sess.run([loss, train_op])
                print('epoch%d-batch%d: loss = %f' % (epoch, batch, l))


if __name__ == '__main__':
    training_data_path = '~/Desktop/cmn.txt'

    # training
    train(training_data_path)

