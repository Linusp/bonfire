import numpy as np


def pad_sequences(seq_list, maxlen, value=0):
    result = np.zeros((len(seq_list), maxlen), dtype=int)
    for idx, seq in enumerate(seq_list):
        length = min(len(seq), maxlen)
        result[idx][:length] = seq[:length]

    return result


def make_batch(data, batch_size=64, loop=False):
    """
    data: list of list, e.g. [[1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e']]
    """
    batch = [[] for _ in data]
    while True:
        for idx in range(len(data[0])):
            if len(batch[0]) == batch_size:
                yield batch
                batch = [[] for _ in data]

            for cnt, row in enumerate(data):
                batch[cnt].append(row[idx])
        if not loop:
            break

    if len(batch[0]) > 0:
        yield batch
