from src.translation_data_processor import DataProcessor
from src.seq2seq_model import EncoderDecoderModel
import tensorflow as tf


def train(data_path, batch_size=50, n_epoch=200):
    """
    load data, train model
    :param data_path:
    :param batch_size:
    :param n_epoch:
    :return: model
    """
    dp = DataProcessor()
    input_ctoi_dict, target_ctoi_dict, input_itoc_dict, target_itoc_dict, encoder_input, decoder_input, decoder_output = dp.load_training_data(data_path)
    out_steps = decoder_output.shape[1]
    out_dims = decoder_output.shape[2]

    encoder_input_placeholder = tf.placeholder(encoder_input.dtype, encoder_input.shape)
    decoder_input_placeholder = tf.placeholder(decoder_input.dtype, decoder_input.shape)
    decoder_output_placeholder = tf.placeholder(decoder_output.dtype, decoder_output.shape)
    dataset = tf.data.Dataset.from_tensor_slices((encoder_input_placeholder, decoder_input_placeholder, decoder_output_placeholder))
    dataset = dataset.batch(batch_size).repeat(n_epoch)
    data_iterator = dataset.make_initializable_iterator()
    encoder_input_tensor, decoder_input_tensor, decoder_output_tensor = data_iterator.get_next()

    model = EncoderDecoderModel(out_steps, out_dims)
    model_output, loss = model(encoder_input_tensor, decoder_input_tensor, decoder_output_tensor)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # init data iterator
        sess.run(data_iterator.initializer, feed_dict={encoder_input_placeholder: encoder_input,
                                                       decoder_input_placeholder: decoder_input,
                                                       decoder_output_placeholder: decoder_output})
        print(sess.run([model_output, loss]))




if __name__ == '__main__':
    training_data_path = '~/Desktop/cmn.txt'

    # training
    train(training_data_path)

