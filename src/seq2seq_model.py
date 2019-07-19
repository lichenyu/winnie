import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, Dense
import numpy as np


'''
One issue to be confronted is the possible variation in both input and output sequence lengths.
To handle that, we can introduce the SOS (Start Of Sequence) and EOS (End Of Sequence) tokens.

The ENCODER loops over the sequence, cell_state initiated by 0.
At each step it takes the encoded representation of a token (word-embedding for example),
and passes it through the recurrent cell, that also gets an input of the state of the sentence representation so far.
When EOS is reached, you collect the final state of your cell (cell_state, hidden_state).

The DECODER shares the same basic architecture, with one added layer (softmax) to predict the new token.
The ENCODER provides a representation of the whole sequence.
Initiate the DECODER with this representation, then send an SOS token to start the generation of the new tokens.
[Training mode]:
  Provide the expected token instead of the predicted decoded token as the new input when generating the next one.
[Inferring mode]:
  Pass the predicted decoded token (consider beam search) as the new input when generating the next one.
Repeat the process until the decoder raises the EOS signal.
'''


class EncoderDecoderModel:
    def __init__(self, out_steps, out_dims, n_units=50):
        self.out_steps = out_steps
        self.out_dims = out_dims
        self.encoder = LSTM(n_units, return_sequences=True, return_state=True)
        self.decoder_rnn = LSTM(n_units, return_sequences=True, return_state=True)
        self.decoder_perceptron = Dense(self.out_dims, activation='softmax')

    def __call__(self, encoder_input, decoder_input, expected_output):
        """
        Training Mode
        :param encoder_input: shape should be (samples, in_steps, in_dims)
        :param decoder_input: shape should be (samples, out_steps, out_dims)
        :param expected_output: shape should be (samples, out_steps, out_dims)
        :return: decoder_output, loss
        """
        encoder_input = tf.cast(encoder_input, tf.float32)
        decoder_input = tf.cast(decoder_input, tf.float32)
        expected_output = tf.cast(expected_output, tf.float32)

        encoder_output, state_h, state_c = self.encoder(encoder_input)
        encoder_state = [state_h, state_c]

        decoder_rnn_output, _, _ = self.decoder_rnn(decoder_input, initial_state=encoder_state)
        decoder_output = self.decoder_perceptron(decoder_rnn_output)

        return decoder_output, self.cal_loss(decoder_output, expected_output)

    def infer(self, encoder_input, sos_token, eos_token):
        """
        Inferring Mode
        :param encoder_input: shape should be (samples, in_steps, in_dims)
        :param sos_token:
        :param eos_token:
        :return:
        """
        encoder_output, state_h, state_c = self.encoder(encoder_input)
        encoder_state = [state_h, state_c]

        init_decoder_input = np.zeros((encoder_input.shape[0], 1, self.out_dims))
        init_decoder_input[:, 0] = sos_token

        input = init_decoder_input
        state = encoder_state
        output = []
        for i in range(self.out_steps):
            decoder_rnn_output, h, c = self.decoder_rnn(input, initial_state=state)
            decoder_output = self.decoder_perceptron(decoder_rnn_output)
            output.append(decoder_output)
            if decoder_output == eos_token:
                break

            input = np.zeros((encoder_input.shape[0], 1, self.out_dims))
            input[:, 0] = decoder_output
            state = [h, c]

        return output

    def cal_loss(self, decoder_perceptron, expected_model_output):
        return tf.losses.mean_squared_error(labels=expected_model_output, predictions=decoder_perceptron)


