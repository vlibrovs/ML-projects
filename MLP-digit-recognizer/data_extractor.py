import numpy as np
from PIL import Image

class DataExtractor:
    def __init__(self):
        self.training_data = "train-images.idx3-ubyte"
        self.training_labels ="train-labels.idx1-ubyte"
        self.testing_data = "t10k-images.idx3-ubyte"
        self.testing_labels = "t10k-labels.idx1-ubyte"

    def extract(self):
        tr_data_file = open(self.training_data, "rb")
        tr_data = np.frombuffer(tr_data_file.read(), np.uint8, offset=16).reshape(-1, 28*28)
        test_data_file = open(self.testing_data, "rb")
        test_data = np.frombuffer(test_data_file.read(), np.uint8, offset=16).reshape(-1, 28*28)
        tr_labels_file = open(self.training_labels, "rb")
        tr_labels = np.frombuffer(tr_labels_file.read(), np.uint8, offset=8)
        test_labels_file = open(self.testing_labels, "rb")
        test_labels = np.frombuffer(test_labels_file.read(), np.uint8, offset=8)

        return tr_data, tr_labels, test_data, test_labels
    
    def get_raw_data(self, id):
        img = Image.open(f"raw_data/img{id}.jpg").load()
        lst = []
        for i in range(28):
            for j in range(28):
                lst.append(img[i, j][0])
        return np.array(lst) / 255
