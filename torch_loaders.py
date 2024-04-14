from torch.utils.data import Dataset

# Dataset then just very simply takes in images and labels and does preprocessing/train test splits
class EMGFrameLoader(Dataset):
    def __init__(self, X, Y, transform=None, target_transform=None, norm=0, train=True, stats=None):
        super(EMGFrameLoader, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.stats = stats
        self.track_stats = True
        self.train = train
        self.X = X
        self.Y = Y
        
        if self.train: self.stats = {}

        if norm  == 1: # standardization
            if train:
                self.mean = self.X.mean()
                self.std = self.X.std()
            else:
                self.mean = stats['mean'] # use training stats 
                self.std = stats['std']
            self.stats['mean'] = self.mean
            self.stats['std'] = self.std
            self.X = (self.X - self.mean)/(self.std + 1.e-12)

        elif norm == -1: # scale between [-1, 1]
            if train:
                self.max = self.X.amax(keepdim=True)
                self.min = self.X.amin(keepdim=True)
            else:
                self.max = stats['max']
                self.min = stats['min']
            self.stats['max'] = self.max
            self.stats['min'] = self.min
            self.X = (self.X - self.min)/(self.max - self.min)*2 - 1

        if transform:
            self.X = transform(self.X)

        self.len = X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        X, Y = self.X[idx], self.Y[idx]
        return X, Y
