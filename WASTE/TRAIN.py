from libraries import *
from utils import *
import os
import argparse
from dataset import train_df, valid_df, TrainDataset
#from efficientnet_pytorch import EfficientNet




def main(args):
    torch.manual_seed(CFG['seed'])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    
    data_transform = {
        "train":transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomChoice([
                    transforms.Pad(padding=10),
                    transforms.CenterCrop(480),
                    transforms.RandomRotation(20),
                    transforms.CenterCrop((576,432)),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1, 
                        saturation=0.1,
                        hue=0.1
                    )
                ]),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
                       
        "val": transforms.Compose([
                transforms.ToPILImage(),   
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    train_dataset = TrainDataset(df=train_df,transform=data_transform["train"])
    val_dataset = TrainDataset(df=valid_df,transform=data_transform["val"])
    
    batch_size = args.batch_size
    #print('Using {} dataloader workers every process'.format(nw))
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True)
                                               #collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True)
                                             #collate_fn=val_dataset.collate_fn)
    #model = models.resnet34(pretrained=True) efficientnet_v2_m
    #model = EfficientNet.from_pretrained('efficientnet-b3')
    #model.fc =  nn.Sequential(
    #                nn.Dropout(0.2),
    #                nn.ReLU(),
    #                nn.Linear(1000, 512),
    #                nn.Linear(124, args.num_classes))
    
    
    #model = timm.create_model(CONFIG.MODEL_NAME, pretrained=True)  
    #model.fc = nn.Sequential(
    #        nn.Dropout(0.2),
    #        nn.ReLU(),
    #        nn.Linear(64, args.num_classes)
    #    )

    #model.to(device)




    
    #model = model.to(device)
    #model = timm.create_model(CONFIG.MODEL_NAME, pretrained=True, in_chans=args.inp_channels)  
    #model.fc = nn.Sequential(
    #        nn.Dropout(0.2),
    #        nn.ReLU(),
    #        nn.Linear(64, args.num_classes)
    #    )
    
    #model.to(device)
    
    model = models.convnext_small(pretrained=True)
    model.head = nn.Sequential(
                     nn.Dropout(0.1),
                     nn.Linear(64, args.num_classes))

    model = model.to(device)            
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=CFG['learning_rate'], momentum=CFG['momentum'])
    
    
    def get_score(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    def train(model, criterion, optimizer, train_loader, val_loader):

        total_train_loss = 0
        total_val_loss = 0

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True

        model.train()
        with tqdm(train_loader, unit='batch', leave=False) as pbar:
            pbar.set_description(f'training')
            for images, idxs in pbar:
                images = images.to(device, non_blocking=True)
                idxs = idxs.to(device, non_blocking=True)
                output = model(images.to(device))

                loss = criterion(output, idxs)
                total_train_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
        model.eval()
        #preds = []
        with tqdm(val_loader, unit='batch', leave=False) as pbar:
            pbar.set_description(f'Validating')
            for images, idxs in pbar:
                images = images.to(device, non_blocking=True)
                idxs = idxs.to(device, non_blocking=True)

                output = model(images.to(device))
                loss = criterion(output, idxs)
                total_val_loss += loss.item()             
        
        train_loss = total_train_loss / len(train_dataset)                  
        val_loss = total_val_loss / len(val_dataset)
        #predictions = np.concatenate(preds)
        return train_loss,val_loss #, predictions
    

        

    #best_score = 0.
    best_loss = 0.00
    for i in range(args.epochs):
        print(f"Epoch {i+1}/{args.epochs}")
        
        #TRAIN::
        train_loss, val_loss =train(model, 
                                  criterion, 
                                  optimizer, 
                                  train_loader,
                                  val_loader)

        #VALIDATE:: #, predictions 
        #val_loss = evaluate(model=model,data_loader=val_loader,device=device)
        #valid_labels = valid_df['Label'].values       
                
        #scoring
        #score = get_score(valid_labels, preds.to(device)).sum()
        print(f'Train loss: {train_loss:.4f} Val loss: {val_loss:.4f}')
        if val_loss<best_loss:
            torch.save(model.state_dict(), "/notebooks/pixels-CLS/RICE/DATA/weights/best_model.pth")
            val_loss=best_loss          
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--lr', type=float, default=0.001)  #5e-4, 
    #parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--inp_channels', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch-size', type=int, default=8)  
        
    opt = parser.parse_args()
    main(opt)