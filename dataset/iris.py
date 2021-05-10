from sklearn.datasets import load_iris

class DataLoader:

    def __init__(self):
        self.dataset = load_iris()

    def load(self, return_X_y=False):
        if return_X_y:
            return self.dataset.data, self.dataset.target
        
        return self.dataset



if __name__ == '__main__':
    
    loader = DataLoader()
    
    dataset = loader.load()

    print(dataset.target_names)
    print(dataset.feature_names)

