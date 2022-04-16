import torch


class Dataset(torch.utils.data.Dataset):
  def __init__(self, list_IDs, features, labels, ages, age, selected=None, posthoc=False):
        self.features = features
        self.labels = labels
        self.ages = ages
        self.list_IDs = list_IDs
        self.age = age
        self.posthoc = posthoc
        self.selected = selected

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.features[ID].values
        y = self.labels[ID].values
        y = y.astype(int)
        ages = self.ages[ID].values
        if self.selected != None:
            feats = self.selected[ID].values

        if self.posthoc and self.age:
            if self.selected != None:
                return X, y, ages, ID, feats
            else:
                return X, y, ages, ID
        elif self.age:
            return X, y, ages
        else:
            return X, y


