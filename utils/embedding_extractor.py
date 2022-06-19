import numpy as np
from .densenet import *
from .dataloader import *

def load_pretrained_model(arch='densenet43',in_features=10574, checkpoint="checkpoint_0199.pth.tar",return_feature = False):
    pretrained, progress = False, False
    kwargs = {'num_classes':2, 'feature_extract':return_feature}
    
    model_dict = {'densenet121':densenet121, 'densenet63':densenet63, 'densenet43':densenet43, 'densenet27':densenet27, 
                  'densenet12':densenet12, 'densenet29':densenet29, 'densenet21':densenet21, 'densenet21_ob':densenet21_ob}
    model_names = model_dict.keys()
    model = model_dict[arch](in_features, pretrained, progress, **kwargs)
    
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # fc layer was not added in case of feature_extract is true.

    checkpoint = torch.load(checkpoint, map_location="cpu")
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
    return model

class MoCoDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        """x: rows are genes and columns are samples"""
        self.X = X
        
    def __getitem__(self, i):
        x = self.X[i,:]
        return x, 0
    
    def __len__(self):
        return len(self.X)
    
def extract_features(model, X):
    # set to eval mode
    model.eval()
    
    val_dataset = MoCoDataset(X)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256, shuffle=False,
        num_workers=2)

    features = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(torch.float32)
            output = model(x)
            features.extend(output.detach().cpu().numpy())
    features = np.asarray(features)
    
    return features