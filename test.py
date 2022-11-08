from libraries import *

import os
import json 
import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm

import torch
from PIL import Image 
from torchvision import transforms 
import matplotlib.pyplot as plt

from dataset import test_df, sample_submission, TestDataset
#from model import EFFNET
#from model import convnext_base as create_model
from utils import CFG, CONFIG


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    
    

    num_classes = 3
    #img_size = 224
    batch_size = args.batch_size
    
    data_transform = transforms.Compose(
                                [transforms.ToPILImage(),
                                 transforms.Resize((384,384)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
 
    
    def inference(model, states, test_loader, device):
        model.to(device)
        tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
        probs = []
        for i, (images) in tk0:
            images = images.to(device)
            avg_preds = []
            model.load_state_dict(states)
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
                avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
            avg_preds = np.mean(avg_preds, axis=0)
            probs.append(avg_preds)
        probs = np.concatenate(probs)
        return probs 
    
    ##########------------################
    # create model
    #model = models.resnet34(pretrained=True)
    #model.fc = nn.Sequential(
        #nn.Dropout(0.1),
        #nn.Linear(model.fc.in_features, args.num_classes)
    #)
    #model = model.to(device)    
    model = timm.create_model(CONFIG.MODEL_NAME, pretrained=True, in_chans=args.inp_channels)  
    model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(124, args.num_classes)
       )
    
    model.to(device)
    
    #model = models.resnet34(pretrained=True)
    #model.fc = nn.Sequential(
        #nn.Dropout(0.1),
        #nn.Linear(model.fc.in_features, args.num_classes))
    #model = model.to(device)    
    
    
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=CFG['learning_rate'], momentum=CFG['momentum'])
    
    # load model weights
    model_weight_path = "/notebooks/pixels-CLS/RICE/DATA/weights/__VIT_BEST.pth"
    #"/notebooks/pixels-CLS/RICE/DATA/weights/_MODEL.pth"
    states = torch.load(model_weight_path)

    #########------------###############
    
     
    # inference
    #model = net
    #states = [torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth') for fold in CFG.trn_fold]
    batch_size = args.batch_size

    test_dataset = TestDataset(df=test_df,transform=data_transform)  
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
                                             #collate_fn=test_dataset.collate_fn)    
    
    
    predictions = inference(model, states, test_loader, device)
    # submission
    submission = pd.DataFrame()
    submission["Image_id"] = sample_submission["Image_id"]
    for i, c in enumerate(sample_submission.columns[1:].to_list()):
        #print(c)
        submission[c] = predictions[:,i]
        submission.to_csv(CONFIG.OUTPUT_DIR+'TRECT.csv', index=False) 
        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--inp_channels', type=int, default=3)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    
    arg = parser.parse_args()


    main(arg)
