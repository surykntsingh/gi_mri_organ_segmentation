import os
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib import animation, rc; rc('animation', html='jshtml', embed_limit=50)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryJaccardIndex
)
from torchmetrics import Dice
from torchmetrics import MetricCollection
from scipy.spatial.distance import directed_hausdorff

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchmetrics
from IPython.display import Image
from skimage import io
import segmentation_models_pytorch as smp
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from IPython.display import display

import torchvision
from torchvision import transforms
from scipy.spatial.distance import directed_hausdorff

from monai.networks.nets import DynUNet
from monai.losses import DiceCELoss

#paths
DATA_PATH = 'data'
MODEL_PATH = 'models'

csv_path = f'{DATA_PATH}/train.csv'
train_data_dir = f'{DATA_PATH}/train'

def preprocess_data(df):
    df[['case','day','_','slice']] = df['id'].str.split('_',n=3,expand=True)
#     df = df.drop('_',axis=1)
    l_bowel_df = df[df["class"]=="large_bowel"].rename(columns={"segmentation":"l_bowel_segmentation"})[['id','l_bowel_segmentation']]
    s_bowel_df = df[df["class"]=="small_bowel"].rename(columns={"segmentation":"s_bowel_segmentation"})[['id','s_bowel_segmentation']]
    stomach_df = df[df["class"]=="stomach"].rename(columns={"segmentation":"stomach_segmentation"})[['id','stomach_segmentation']]
    
    df = df.merge(l_bowel_df, on='id', how='left')
    df = df.merge(s_bowel_df, on='id', how='left')
    df = df.merge(stomach_df, on='id', how='left')
    df['l_bowel_flag'] = df['l_bowel_segmentation'].notna()
    df['s_bowel_flag'] = df['s_bowel_segmentation'].notna()
    df['stomach_flag'] = df['stomach_segmentation'].notna()
    df['num_segments'] = df["l_bowel_flag"].astype(int)+df["s_bowel_flag"].astype(int)+df["stomach_flag"].astype(int)

    df = df.drop(columns=['_','segmentation','class'], axis=1).drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


def get_files_data_df(data_dir):
    idx=len(data_dir.split('/'))
    all_files=glob(f'{data_dir}/*/*/scans/*.png')
    data_df = pd.DataFrame(all_files,columns=['full_path'])
    data_df['path_arr'] = data_df['full_path'].str.split('/') 
    data_df['file_name'] = data_df['path_arr'].str[-1]
    data_df['case_id'] = data_df['path_arr'].str[idx]
    data_df['day_id'] = data_df['path_arr'].str[idx+1].str.split('_').str[1]
    # data_df['slice_arr'] = data_df['file_name'].str.split('_',n=6,expand=True)
    data_df[['slice_lit','slice_id','slice_height_px','slice_width_px','spacing_height_px','spacing_width_px']] = data_df['file_name'].str.split('_',n=5,expand=True)
    data_df['spacing_width_px'] = data_df['spacing_width_px'].str[:-4]

    data_df['id'] = data_df[['case_id','day_id','slice_lit','slice_id']].agg('_'.join,axis=1)
    data_df = data_df.drop(['path_arr','slice_lit'], axis=1)
    data_df[['slice_height_px','slice_width_px']] = data_df[['slice_height_px','slice_width_px']].astype(int)
    data_df[['spacing_height_px','spacing_width_px']] = data_df[['spacing_height_px','spacing_width_px']].astype(float)
    return data_df



class ImageUtils():
    
    @staticmethod
    def decode_rle(mask, shape, color=1):
        '''decodes rle encoded mask and returns image of size shape'''
        s = np.array(mask.split(), dtype=int)
        starts = s[0::2] - 1
        lengths = s[1::2]
        ends = starts + lengths
        if len(shape)==3:
            h, w, d = shape
            img = np.zeros((h * w, d), dtype=np.float32)
        else:
            h, w = shape
            img = np.zeros((h * w,), dtype=np.float32)
            
        for lo, hi in zip(starts, ends):
            img[lo : hi] = color
        return img.reshape(shape)
    
    # ref.: https://www.kaggle.com/stainsby/fast-tested-rle
    @staticmethod
    def rle_encode(img, img_shape):
        '''encodes segment mask to rle encoded string '''
        img = cv2.resize(img, img_shape)
        pixels = img.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
    
    @staticmethod
    def open_gray16(_path, img_shape=None, normalize=True, rgb=True):
        '''loads image from image path and returns normalized and resized image'''
        img = cv2.imread(_path, cv2.IMREAD_ANYDEPTH)
        # print(img.shape)
        if img_shape:
            img = cv2.resize(img, img_shape)
            
        if normalize:
            img = img/np.amax(img)
            
        if rgb:
            return np.tile(np.expand_dims(img, axis=-1), 3)
        else:
            return img
    
    @staticmethod
    def get_segment_masks(segment_mask_map, original_img_shape, new_img_shape=None):
        '''returns segment masks for each segment type'''
        segment_map = {}
        for segment_type in segment_mask_map:
            if segment_mask_map[segment_type]:
                segment_map[segment_type] = ImageUtils.decode_rle(segment_mask_map[segment_type], shape=original_img_shape, color=1)
            else:
                segment_map[segment_type] = np.zeros(original_img_shape, dtype=np.float32)
        return segment_map
            
    @staticmethod
    def get_overlay(img_path, mask_map, img_shape, _alpha=0.999, _beta=0.35, _gamma=0):
        '''Overlays segment masks on top of images and returns overlayed image'''
        _img = ImageUtils.open_gray16(img_path, rgb=True)
        _img = ((_img-_img.min())/(_img.max()-_img.min())).astype(np.float32)
        
        segment_array = ImageUtils.get_segment_masks(mask_map, img_shape)
        _seg_rgb = np.stack(list(segment_array.values()), axis=-1).astype(np.float32)
        
        seg_overlay = cv2.addWeighted(src1=_img, alpha=_alpha, 
                                      src2=_seg_rgb, beta=_beta, gamma=_gamma)
        seg_overlay = seg_overlay/np.amax(seg_overlay)
        return seg_overlay
    
    @staticmethod
    def display_original(img_path, example):
        '''Displays original image'''
        display(example.to_frame())
        plt.figure(figsize=(12,12))
        plt.imshow(ImageUtils.open_gray16(example.full_path), cmap="gray")
        plt.title(f"Original Grayscale Image For ID: {example.id}", fontweight="bold")
        plt.axis(False)
        plt.show()
      
    @staticmethod
    def display_masks(img_path, example):
        '''Displays segment masks'''
        print(f"... Binary segment mask ...")
        plt.figure(figsize=(20,10))
        mask_types = ('l_bowel_segmentation', 's_bowel_segmentation','stomach_segmentation')
        mask_str_map = {mask_type: example[mask_type] if pd.notna(example[mask_type]) else None for mask_type in mask_types}
        img_shape = (example.slice_height_px, example.slice_width_px)
        segment_map = ImageUtils.get_segment_masks(mask_str_map, img_shape)
        i=0
        for segment_type in segment_map:
            plt.subplot(1,3,i+1)
            plt.imshow(segment_map[segment_type])
            plt.title(f"RLE segment mask For {segment_type} Segmentation", fontweight="bold")
            plt.axis(False)
            i+=1
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def display_masked_overlay(img_path, example):
        '''displayed masked overlayed image'''
        mask_types = ('l_bowel_segmentation', 's_bowel_segmentation','stomach_segmentation')
        mask_str_map = {mask_type: example[mask_type] if pd.notna(example[mask_type]) else None for mask_type in mask_types}
        
        seg_overlay = ImageUtils.get_overlay(example.full_path, mask_str_map, img_shape=(example.slice_height_px, example.slice_width_px))
#         print(seg_overlay, type(seg_overlay))
        plt.figure(figsize=(12,12))
        plt.imshow(seg_overlay)
        plt.title(f"Segmentation Overlay For ID: {example.id}", fontweight="bold")
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel Segmentation Map", "Small Bowel Segmentation Map", "Stomach Segmentation Map"]
        plt.legend(handles,labels)
        plt.axis(False)
        plt.show()
    
    @staticmethod
    def create_animation(case_id, day_id, df):
        '''Created animation from an image stack'''
        select_cols = ['full_path','l_bowel_segmentation', 's_bowel_segmentation',
                       'stomach_segmentation', 'slice_height_px', 'slice_width_px']
        filtered_df = df[(df.case_id==case_id) & (df.day_id==day_id)][select_cols]
        
        
        paths  = filtered_df.full_path.tolist()
        lb_rles  = filtered_df.l_bowel_segmentation.tolist()
        sb_rles  = filtered_df.s_bowel_segmentation.tolist()
        st_rles  = filtered_df.stomach_segmentation.tolist()
        slice_ws = filtered_df.slice_width_px.tolist()
        slice_hs = filtered_df.slice_height_px.tolist()
        
        animation_arr = np.stack([
            ImageUtils.get_overlay(img_path=_f, mask_map={k: v if pd.notna(v) else None for k,v in {'l_bowel_segmentation':_lb,'s_bowel_segmentation': _sb,'stomach_segmentation': _st}.items()}, img_shape=(_w, _h)) \
            for _f, _lb, _sb, _st, _w, _h in \
            zip(paths, lb_rles, sb_rles, st_rles, slice_ws, slice_hs)
        ], axis=0)
    
        fig = plt.figure(figsize=(8,8))

        plt.axis('off')
        im = plt.imshow(animation_arr[0])
        plt.title(f"3D Animation for Case {case_id} on Day {day_id}", fontweight="bold")

        anim = animation.FuncAnimation(fig, lambda x: im.set_array(animation_arr[x]), frames = animation_arr.shape[0], interval = 1000//12)
        plt.close()
        return anim


class UnetDataset(Dataset):
    
    def __init__(self, df, img_shape, segment_type=None, predict = False):
        self.df = df
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.img_shape = img_shape
        self.segment_type = segment_type
        self.predict = predict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img = ImageUtils.open_gray16(item.full_path, self.img_shape)
        img_tensor = self.transform(img).to(torch.float32)
        if self.predict:
            return img_tensor
        else:
            mask_str_map = {self.segment_type: item[self.segment_type] if pd.notna(item[self.segment_type]) else None}
            img_shape = (item.slice_height_px, item.slice_width_px)
            segment_mask_map = ImageUtils.get_segment_masks(mask_str_map, img_shape)
            segment_mask_map = {segment_type: cv2.resize(mask, self.img_shape) for segment_type,mask in segment_mask_map.items()}

            segment_tensor = self.transform(segment_mask_map[self.segment_type]).to(torch.float32)
            return img_tensor, segment_tensor

class NnUNetModel(pl.LightningModule):
    def __init__(self, lr=1e-4, threshold=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = DynUNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            kernel_size=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2],
            norm_name="INSTANCE",
            deep_supervision=False,
        )

        self.loss_fn = DiceCELoss(sigmoid=True)
        self.threshold = threshold
        self.lr = lr

        metrics = MetricCollection({
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "f1": BinaryF1Score(threshold=threshold),
            "iou": BinaryJaccardIndex(threshold=threshold),
            "dice": Dice(threshold=threshold),
        })

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.test_hausdorff = []

    def forward(self, x):
        return self.model(x)

    def model_step(self, batch):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        loss = self.loss_fn(probs, y)
        return loss, probs, y

    def training_step(self, batch):
        loss, preds, targets = self.model_step(batch)
        self.train_metrics.update(preds, targets.int())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch):
        loss, preds, targets = self.model_step(batch)
        self.val_metrics.update(preds, targets.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics,on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def test_step(self, batch):
        loss, preds, targets = self.model_step(batch)
        self.test_metrics.update(preds, targets.int())
        hd = self.hausdorff_distance(preds, targets)
        self.test_hausdorff.append(hd)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        hausdorff_avg = np.mean(self.test_hausdorff)
        self.log("test_hausdorff", hausdorff_avg, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()
        self.test_hausdorff.clear()

    def hausdorff_distance(self, pred, target):
        pred = (pred > 0.5).squeeze().cpu().numpy()
        target = target.squeeze().cpu().numpy()
        pred_pts = torch.nonzero(torch.tensor(pred)).numpy()
        target_pts = torch.nonzero(torch.tensor(target)).numpy()
        if len(pred_pts) == 0 or len(target_pts) == 0:
            return 0.0
        return max(
            directed_hausdorff(pred_pts, target_pts)[0],
            directed_hausdorff(target_pts, pred_pts)[0]
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class UnetDataModule(pl.LightningDataModule):
    def __init__(self, data_df, img_shape, segment_type, split_frac, batch_size=64):
        super().__init__()
        self.data_df = data_df
        self.image_shape = img_shape
        self.batch_size = batch_size
        self.segment_type = segment_type
        self.split_frac = split_frac

    def setup(self, stage=None):
        ds = UnetDataset(self.data_df, img_shape=self.image_shape, segment_type=self.segment_type)
        self.train_ds, self.val_ds, self.test_ds = random_split(ds, self.split_frac)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

class UnetModelTrainer():
    
    def __init__(self, data_df, batch_size, dir_path,img_shape,segment_type,
                 max_epochs=10,split_frac=[0.65, 0.2, 0.15]):
        self.df=data_df
        self.max_epochs = max_epochs
        self.split_frac = split_frac
        self.model = NnUNetModel()
        self.datamodule = UnetDataModule(data_df, img_shape, segment_type, split_frac)
        # self.datamodule.setup()
        self.dir_path = f'{dir_path}/{segment_type}'
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path, exist_ok=True)
        
        self.checkpoint_path = f'{self.dir_path}/best_model.ckpt'
        pl.seed_everything(42)
        
    def train(self, fast_dev_run=False):
        # checkpoint_callback = ModelCheckpoint(monitor = "val_f1_score", mode= 'max', 
        #                                       filename='best_model', dirpath=self.dir_path,
        #                                   save_top_k = 1, save_weights_only=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.dir_path,  # Directory to save checkpoints
            filename="best_model",  # Naming convention
            monitor="val_loss",  # Metric to monitor for saving best checkpoints
            mode="min",  # Whether to minimize or maximize the monitored metric
            save_top_k=1,  # Number of best checkpoints to keep
            save_last=True  # Save the last checkpoint regardless of the monitored metric
        )
        
        early_stop_callback = EarlyStopping(monitor= "val_loss", min_delta=1e-4, patience = 3, verbose=True, mode="min")
        
        swa_callback = StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=0.001, annealing_epochs=5, annealing_strategy='cos')

        self.trainer = pl.Trainer(
            max_epochs= self.max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback, swa_callback],
            accelerator='gpu',
            devices=[0,1,2],
            strategy='ddp',
            enable_progress_bar=True,
            log_every_n_steps = 2,
            fast_dev_run=fast_dev_run
        )

        train_metrics = self.trainer.fit(
            self.model, datamodule = self.datamodule
        )

        # run test dataset
        test_metrics = self.trainer.test(self.model, datamodule = self.datamodule, 
                                         ckpt_path=self.checkpoint_path, verbose=True)
        pprint(test_metrics)
        metrics = {
            'train':train_metrics,
            'test':test_metrics
        }
        print('model training complete')
        pprint(metrics)
        
    def evaluate_visual(self, threshold=0.5):
        self.model = UNetModel.load_from_checkpoint('models/models/l_bowel_segmentation/best_model-v3.ckpt')
        # self.model = self.model.load_from_checkpoint(self.checkpoint_path)
        batch = next(iter(self.datamodule.test_dataloader()))
        with torch.no_grad():
            self.model.eval()
            logits = self.model(batch[0])
        pr_masks = (logits.sigmoid() > threshold).float()
        print(f'evaluating and visualizing for segment_type: {self.segment_type}')
        
        for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image.detach().cpu().numpy().transpose(1, 2, 0))
            plt.title("Image")
            plt.axis("off")
            _gt_mask=gt_mask.squeeze().detach().cpu().numpy()
            _pr_mask=pr_mask.squeeze().numpy()
            hausdorff_dist=directed_hausdorff(_gt_mask,_pr_mask)[0]
            plt.subplot(1, 3, 2)
            plt.imshow(_gt_mask) 
            plt.title("Ground truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(_pr_mask) 
            plt.title("Prediction")
            plt.axis("off")
            print(f'hausdorff_dist:: {hausdorff_dist}')
            plt.show()


if __name__ == '__main__':
    data_df = pd.read_csv(csv_path)
    processed_data_df = preprocess_data(data_df)
    files_data_df = get_files_data_df(train_data_dir)
    data_meta_df = processed_data_df.merge(files_data_df, on="id", how="inner")

    segment_types = ('l_bowel_segmentation', 's_bowel_segmentation','stomach_segmentation')
    batch_size = 128
    image_shape = (128,128)
    dir_path = f'{MODEL_PATH}/models/nn_unet/mgpu/0'
    max_epochs = 50
    model_trainers = {
        segment_type: UnetModelTrainer(data_meta_df, batch_size, dir_path, 
                                       image_shape, segment_type, 
                                       max_epochs=max_epochs)
        for segment_type in segment_types
    }


    seg_id=0
    print(f'training model for {segment_types[seg_id]}')
    model_trainers[segment_types[seg_id]].train()
    print('Finished!!')

    seg_id=1
    print(f'training model for {segment_types[seg_id]}')
    model_trainers[segment_types[seg_id]].train()
    print('Finished!!')

    seg_id=2
    print(f'training model for {segment_types[seg_id]}')
    model_trainers[segment_types[seg_id]].train()
    print('Finished!!')
