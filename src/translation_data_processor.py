import pandas as pd
import numpy as np


class DataProcessor:
    def __init__(self):
        return

    def load_training_data(self, data_path, sep='\t', samples=2000):
        df = pd.read_csv(data_path, sep=sep, header=None, names=['input', 'target']).iloc[0:samples, :]
        assert (isinstance(df, pd.DataFrame))

        # target每句前后增加'\t'、'\n'，作为起始、终止标志
        df['target'] = df['target'].apply(lambda x: '\t' + x + '\n')

        input_sentence_list = df['input'].tolist()
        target_sentence_list = df['target'].tolist()
        input_char_list = sorted(list({c for char_list in [list(sen) for sen in df['input']] for c in char_list}))
        target_char_list = sorted(list({c for char_list in [list(sen) for sen in df['target']] for c in char_list}))
        input_ctoi_dict = {c: i for i, c in enumerate(input_char_list)}
        input_itoc_dict = {i: c for i, c in enumerate(input_char_list)}
        target_ctoi_dict = {c: i for i, c in enumerate(target_char_list)}
        target_itoc_dict = {i: c for i, c in enumerate(target_char_list)}

        in_steps = max([len(s) for s in input_sentence_list])
        out_steps = max([len(s) for s in target_sentence_list])
        # simply use one-hot
        in_dims = len(input_char_list)
        out_dims = len(target_char_list)

        encoder_input = np.zeros((samples, in_steps, in_dims))
        decoder_input = np.zeros((samples, out_steps, out_dims))
        decoder_output = np.zeros((samples, out_steps, out_dims))
        for sample, sen in enumerate(input_sentence_list):
            for step, char in enumerate(sen):
                encoder_input[sample, step, input_ctoi_dict[char]] = 1
        for sample, sen in enumerate(target_sentence_list):
            for step, char in enumerate(sen):
                # start with SOS, end with EOS
                decoder_input[sample, step, target_ctoi_dict[char]] = 1
                if step > 0:
                    decoder_output[sample, step - 1, target_ctoi_dict[char]] = 1

        return input_ctoi_dict, target_ctoi_dict, input_itoc_dict, target_itoc_dict, encoder_input, decoder_input, decoder_output


if __name__ == '__main__':
    dp = DataProcessor()
    out = dp.load_training_data('~/Desktop/cmn.txt')
    print(out)
