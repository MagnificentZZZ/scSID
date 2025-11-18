from torch.utils import data
class mydataSet(data.dataset.Dataset):
    def __init__(self, data, label, sid):
        super(mydataSet, self).__init__()
        self.data = data
        self.label = label
        self.sid = sid
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        z = self.sid[index]    
        return x, y, z
    def __len__(self):
        return len(self.data)