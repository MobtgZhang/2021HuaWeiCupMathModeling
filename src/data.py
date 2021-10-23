import pandas as pd
# prepare the dataset
from sklearn.preprocessing import MinMaxScaler
class MolDataSet:
    def __init__(self, train_x,train_y=None,transform:MinMaxScaler=None, target_transform:MinMaxScaler=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train_x = train_x
        self.train_y = train_y

        if self.train_y is not None:
            if self.transform:
                self.transform.fit(self.train_x)
                self.target_transform.fit(self.train_y)
                self.train_x = self.transform.transform(self.train_x)
                self.train_y =self.target_transform.transform(self.train_y)
        else:
            self.transform.fit(self.train_x)
            if self.transform:
                self.train_x = self.transform.transform(self.train_x)

    def __getitem__(self, index):
        if self.train_y is not None:
            return self.train_x[index], self.train_y[index]
        else:
            return self.train_x[index]

    def __len__(self):
        return len(self.train_x)

    def get_data(self):
        if self.train_y is not None:
            return self.train_x, self.train_y
        else:
            return self.train_y
class ADMETDataSet:
    def __init__(self, train_x, train_y=None, transform: MinMaxScaler = None):
        self.transform = transform
        self.train_x = train_x
        self.train_y = train_y
        if self.transform:
            self.transform.fit(self.train_x)
            self.train_x = self.transform.transform(self.train_x)

    def __getitem__(self, index):
        if self.train_y is not None:
            return self.train_x[index], self.train_y[index]
        else:
            return self.train_x[index]

    def __len__(self):
        return len(self.train_x)

    def get_data(self):
        if self.train_y is not None:
            return self.train_x, self.train_y
        else:
            return self.train_y

# data loader processing
class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.start = 0
        self.k_numbers = 0
        self.length = len(self.dataset)
        self.batch_size = batch_size
        self.num = int(self.length/self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.k_numbers > self.num:
            raise StopIteration
        else:
            start = self.start + self.k_numbers * self.batch_size
            end = self.start + (self.k_numbers + 1) * self.batch_size
            self.k_numbers += 1
            return self.dataset[start:end]

    def clear(self):
        self.start = 0
        self.k_numbers = 0
