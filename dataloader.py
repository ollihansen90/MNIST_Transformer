import torch

class Dataloader():
    def __init__(self, dataset, num_classes=10):
        self.dataset = dataset
        self.N = len(dataset)
        self.num_classes = num_classes

    def getbatch(self, n=1):
        sample = self.dataset[torch.randint(self.N, (1,))]
        batch = sample["image"].unsqueeze(0)
        labels = torch.tensor(sample["label"]).unsqueeze(0)
        for _ in range(1, n):
            sample = self.dataset[torch.randint(self.N, (1,))]
            batch = torch.cat((batch, sample["image"].unsqueeze(0)), dim=0)
            labels = torch.cat((labels, torch.tensor(sample["label"]).unsqueeze(0)), dim=0)

        labelsout = torch.zeros((n,self.num_classes))
        labelsout[list(range(n)), labels-1] = 1
        return batch, labelsout

