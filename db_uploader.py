from pymongo import MongoClient

TASK = 27


def get_data(delete_index=True):
    db = MongoClient().get_database('bigdata')
    data = db.bigdata.find({"task": TASK})
    training = []
    test = []
    for row in data:
        del row['_id']
        del row['task']
        if delete_index:
            del row['i']
        if row.get('y'):
            training.append(row)
        else:
            test.append(row)
    return training, test


if __name__ == '__main__':
    training_data, test_data = get_data()
