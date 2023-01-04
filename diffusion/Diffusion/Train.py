
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler
from datasets import load_dataset
import matplotlib.pyplot as plt

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # datasetCJ
    dataset=load_dataset("Team-PIXEL/rendered-wikipedia-english", split="train", streaming=True)
    dataset=dataset.with_format("torch")
    dataset=dataset.remove_columns(['num_patches'])
    def transformation(examples):
        transform=transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.functional.crop(0,0,16,1024),
            #transforms.Resize([16,224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        examples["pixel_values"] = [transform(transforms.functional.crop(image.convert("RGB"),0,0,16,1024)) for image in examples["pixel_values"]]
        return examples
    dataset = dataset.map(transformation, batched=True, batch_size=modelConfig["batch_size"])
    #dataset=trainsform(traindata)
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"])
    #for labels, images in enumerate(dataset):
       # images=images["pixel_values"]
      #  save_image(images, os.path.join(
     #       modelConfig["sampled_dir"],  "img.png"), nrow=modelConfig["nrow"])
    #    return 
    #print(dataloader)
    #for labels in tqdm(dataloader, dynamic_ncols=True):
    #   print(labels["pixel_values"],type(labels['pixel_values']))
    #    break
    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    #for e in range(modelConfig["epoch"]):
    cnt=0
    e=0

    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for lables, images in enumerate(dataloader):
                # train
                #print(images, labels)
            images=images['pixel_values']
                #print(images.shape)
            optimizer.zero_grad()
            x_0 = images.to(device)
            loss = trainer(x_0).sum() / 1000.

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), modelConfig["grad_clip"])
            optimizer.step()
        #    tqdmDataLoader.set_postfix(ordered_dict={
         #       "epoch": e,
          #      "loss: ": loss.item(),
           #     "img shape: ": x_0.shape,
            #    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
            #})
            cnt+=1
            if cnt==1000:
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                #print("epoch:{}, loss:{}, img shape:{}, LR:{}".format(e,loss.item(),x_0.shape, optimizer.state_dict()['param_groups'][0]["lr"]))
                cnt=0
                e+=1
                warmUpScheduler.step()
                if e%30==0:
                    torch.save(net_model.state_dict(), os.path.join(
                        modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_1.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 16, 1024], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])