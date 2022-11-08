import os
import argparse
#!pip install efficientnet_pytorch


import torch 
import torch.optim as optim
#import albumentations as A
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,models
#from albumentations import (
    #Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    #RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    #IAAAdditiveGaussianNoise, Transpose
 #   )
#from albumentations.pytorch import ToTensorV2
#from albumentations import ImageOnlyTransform
 

from dataset import train_df, valid_df, TrainDataset
#from efficientnet_pytorch import EfficientNet
#from model import EFFNET
from model import convnext_base as create_model
from utils import CFG, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate




def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("/notebooks/pixels-CLS/RICE/DATA/weights") is False: 
        os.makedirs("/notebooks/pixels-CLS/RICE/DATA/weights")

    tb_writer = SummaryWriter()


    #img_size = 224
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
                transforms.Resize((512,640)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
                       
        "val": transforms.Compose([
                transforms.ToPILImage(),   
                transforms.Resize((512,640)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
                                   
                                   
          
 
       

    train_dataset = TrainDataset(df=train_df,transform=data_transform["train"])
    val_dataset = TrainDataset(df=valid_df,transform=data_transform["val"])
    
    
    
    

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("TRAINING{}".format(name))
    
    #using efficientnet model based transfer learning
    #model = EFFNET().to(device)



    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "/notebooks/pixels-CLS/RICE/DATA/weights/best_model.pth")
            best_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)  #5e-4, 
    parser.add_argument('--wd', type=float, default=0.001)


    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, 
                        default= "/notebooks/pixels-CLS/RICE/DATA/weights/convnext_base_22k_224.pth",
                        help='initial weights path')
                        #"/notebooks/pixels-CLS/RICE/DATA/weights/resnet34-b627a593.pth",
                        #'/notebooks/pixels-CLS/RICE/DATA/weights/convnext_tiny_1k_224_ema.pth',
                        
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
