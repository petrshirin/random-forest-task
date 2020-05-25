import pickle
from db_uploader import get_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from learn_forest import Tokenizer


def predict():
    tr, test = get_data(False)
    f = open('forest.pkl', 'rb')
    pik = pickle.loads(f.read())
    f.close()
    forest = pik[0]
    tokenizer = Tokenizer([])
    tokenizer.tokens = pik[1]
    result = open('result', 'w')
    for test_var in test:
        i = test_var['i']
        del test_var['i']

        x_test = []
        for key, value in test_var.items():
            x_test.append(tokenizer.get_token_by_value(value))
        res = tokenizer.get_key_by_token(forest.predict(np.array([x_test]))[0])
        result.write(f'{int(i)} {res}\n')
        print(int(i), res)
    result.close()


if __name__ == '__main__':
    predict()







