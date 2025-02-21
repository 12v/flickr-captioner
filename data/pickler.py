import pickle


def index_pickle(file_path):
    """Indexes a pickle file by storing byte positions of each object."""
    offsets = []
    i = 0
    with open(file_path, "rb") as f:
        while True:
            pos = f.tell()  # Get current byte position
            try:
                pickle.load(f)
                offsets.append(pos)
                print(i, pos)
                i += 1
            except EOFError:
                break
    return offsets


def fast_seek(file_path, offsets, n):
    """Loads the nth object using a precomputed index of byte offsets."""
    with open(file_path, "rb") as f:
        f.seek(offsets[n])
        obj = pickle.load(f)
        return obj
