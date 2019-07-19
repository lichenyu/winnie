from src.data_processor import DataProcessor
from src.seq2seq_model import EncoderDecoderModel
import tensorflow as tf


'''
load data, train model
return model
'''
def train(data_path):
    dp = DataProcessor()
    input_ctoi_dict, target_ctoi_dict, input_itoc_dict, target_itoc_dict, encoder_input, decoder_input, decoder_output = dp.load_training_data(data_path)

    model = EncoderDecoderModel(decoder_output.shape[1], decoder_output.shape[2])
    model(tf.convert_to_tensor(encoder_input), tf.convert_to_tensor(decoder_input), tf.convert_to_tensor(decoder_output))

    return model


if __name__ == '__main__':
    training_data_path = '~/Desktop/cmn.txt'

    # training
    train(training_data_path)

