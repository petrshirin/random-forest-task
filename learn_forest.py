from sklearn.ensemble import RandomForestClassifier
from db_uploader import get_data
import numpy as np
import pickle


class Tokenizer(object):

    def __init__(self, data: list):
        self.tokens = {}
        self.split_data(data)

    def split_data(self, data: list):
        i = 1
        for row in data:
            for key, value in row.items():
                if self.tokens.get(value) is None:
                    self.tokens[value] = i
                    i += 1

    def get_key_by_token(self, your_token: int):
        for key, token in self.tokens.items():
            if token == your_token:
                return key

    def get_token_by_value(self, your_value: str):
        return self.tokens[your_value]


def generate_xy_train(tokenizer: Tokenizer, data: list, count_param=20):
    x_train = []
    y_train = []
    for row in data:
        x_train_row = []
        for var, value in row.items():
            if var == 'y':
                y_train.append(tokenizer.get_token_by_value(value))
                continue
            else:
                x_train_row.append(tokenizer.get_token_by_value(value))
        x_train.append(x_train_row)
    return x_train, y_train


def generate_forest():
    training, test = get_data()
    # training = training[:int(len(training)*0.8)]
    tokenizer = Tokenizer(training + test)
    x_train, y_train = generate_xy_train(tokenizer, training)
    x = np.array(x_train)
    forest = RandomForestClassifier()
    forest.fit(x, y_train)

    with open('forest.pkl', 'wb') as f_forest:
        pickle.dump((forest, tokenizer.tokens,), f_forest)


if __name__ == '__main__':
    generate_forest()
