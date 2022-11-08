from libraries import *
#from train import *
#from dataset import train_df, valid_df, TrainDataset


CFG = {
    'epochs': 25,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'momentum': 0.9,
    'test_size': 0.05,
    'seed': 1,}



class FOLD_CFG:
    LR = 1e-4
    SEED = 2020
    SPLITS = 5
    MODEL_NAME = 'efficientnet-b3'

class CONFIG:
    MODEL_NAME = "vit_base_patch32_224"  #"convnext_nano"#"vit_tiny_patch16_224"
    
    #"vit_small_patch16_384"
    #'swin_tiny_patch4_window7_224'
                    #'swin_base_patch4_window12_224'  #'convnext_nano' swin_base_patch4_window7_224
    OUTPUT_DIR = "/notebooks/pixels-CLS/RICE/DATA/"
    

## function
def create_params_to_update(net):
    params_to_update_1 = []
    update_params_name_1 = ['_fc.weight', '_fc.bias']

    for name, param in net.named_parameters():
        if name in update_params_name_1:
            param.requires_grad = True
            params_to_update_1.append(param)
            #print("{} 1".format(name))
        else:
            param.requires_grad = False
            #print(name)

    params_to_update = [{'params': params_to_update_1, 'lr': LR}]
    return params_to_update


def adjust_learning_rate(optimizer, epoch):
    lr = FOLD_CFG.LR * (0.1**(epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr
 

