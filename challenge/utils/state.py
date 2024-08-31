import torch

def save_model_state(model, path):
    torch.save(model.state_dict(), path)
    
def load_model_state(path):
    torch.load(path)