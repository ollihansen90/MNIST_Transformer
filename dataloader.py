import torch

class Dataloader():
    def __init__(self, dataset, labels=list(range(10))):
        self.dataset = dataset
        self.N = len(dataset)
        self.labels = labels
        self.num_classes = len(labels)

    def getbatch(self, n=1):
        sample = self.dataset[torch.randint(self.N, (1,))]
        batch = sample["image"].unsqueeze(0)
        #labels = torch.tensor(sample["label"]).unsqueeze(0)
        labels = [sample["label"]]
        for _ in range(1, n):
            sample = self.dataset[torch.randint(self.N, (1,))]
            batch = torch.cat((batch, sample["image"].unsqueeze(0)), dim=0)
            #labels = torch.cat((labels, torch.tensor(sample["label"]).unsqueeze(0)), dim=0)
            labels.append(sample["label"])
        
        #labels = torch.tensor(labels)
        labelsout = torch.zeros((n, self.num_classes))
        for i, label in enumerate(labels):
            idx = self.labels.index(label)
            labelsout[i, idx] = 1
        #print(labelsout)
        return batch, labelsout

