from torch.utils.data import Dataset as PytorchDataset
import os
import torch
from torchvision import transforms
from .transformations import flow2rgb
import numpy as np
import PIL
from skimage import io
import png
import cv2

'''
class PairDataset(PytorchDataset):
    torch.nn.functional.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False)

    def __len__(self):
        if self.return_indices:
            return self.to_frame - self.from_frame - 1
        else:
            return self.to_frame - self.from_frame - 1

    def __getitem__(self, idx):
        pass
'''

class KittiDataset(PytorchDataset):
    def __init__(self, imgs_dir, flow_dir=None, preload=False):

        if flow_dir is not None:
            self.flow_dir = flow_dir
            self.flow_filenames = os.listdir(self.flow_dir)
        else:
            self.flow_filenames = None

        self.imgs_dir = imgs_dir
        self.images_filenames = os.listdir(self.imgs_dir)

        self.dtype = torch.float32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preload = preload

        self.width = 640
        self.height = 640

        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor()])

        self.num_seq = 0
        self.seq_length = 0

        self.el_min_id = 2**30
        self.el_max_id = -1

        if self.preload:
            self.imgs = []
            self.flows = []

        for img_fn in self.images_filenames:
            img_fn_splitted = img_fn.split('_')
            seq_id = int(img_fn_splitted[0])
            el_id = int(img_fn_splitted[1].split('.')[0])

            if seq_id+1 > self.num_seq:
                self.num_seq = seq_id+1

            if el_id < self.el_min_id:
                self.el_min_id = el_id

            if el_id > self.el_max_id:
                self.el_max_id = el_id

            if self.preload:
                img_fp = os.path.join(self.imgs_dir, img_fn)
                self.imgs.append(self.read_rgb(img_fp, device='cpu'))

                if self.flow_filenames is not None:
                    flow_fp = os.path.join(self.flow_dir, self.flow_filenames[seq_id])
                    self.flows.append(self.read_flow(flow_fp, device='cpu'))

        self.seq_length = self.el_max_id-self.el_min_id+1
        print('self.num_seq', self.num_seq)
        print('self.seq_length', self.seq_length)


    def __len__(self):

        return self.num_seq * (self.seq_length-1)

    def __getitem__(self, id):

        seq_id = id // (self.seq_length-1)
        el_id = id % (self.seq_length-1)

        img1_id = seq_id*self.seq_length + el_id
        img2_id = seq_id*self.seq_length + el_id + 1

        if self.preload:
            img1 = self.imgs[img1_id].to(self.device)
            img2 = self.imgs[img2_id].to(self.device)
        else:
            img1_fn = os.path.join(self.imgs_dir, self.images_filenames[img1_id])
            img2_fn = os.path.join(self.imgs_dir, self.images_filenames[img2_id])

            img1 = self.read_rgb(img1_fn, device=self.device)
            img2 = self.read_rgb(img2_fn, device=self.device)

        imgpair = torch.cat((img1, img2), dim=0)

        if self.flow_filenames is not None:

            if self.preload:
                flow_uv, flow_valid = self.flows[seq_id]
                flow_uv = flow_uv.to(self.device)
                flow_valid = flow_valid.to(self.device)
            else:
                flow_fn = os.path.join(self.flow_dir, self.flow_filenames[seq_id])
                flow_uv, flow_valid = self.read_flow(flow_fn, device=self.device)

            # flow_rgb = torch.from_numpy(flow2rgb(flow_uv))
            # torch.float 3xHxW

            #pil_transform = transforms.ToPILImage()
            #pil_transform(flow_rgb).show()
            return imgpair, flow_uv, flow_valid
        else:
            return imgpair

    def read_flow(self, flow_fn, device):

        flow = cv2.imread(flow_fn, cv2.IMREAD_UNCHANGED)
        # numpy.ndarray: HxWx3

        flow_valid = torch.from_numpy(flow[:, :, 0].astype(np.bool)).to(device)
        # torch.bool: HxW

        flow_uv = (torch.from_numpy(flow[:, :, 1:].astype(np.int32)).to(device).permute(2, 0, 1) - 2**15) / 64.0
        # torch.float32: 2xHxW

        return flow_uv, flow_valid

    def read_rgb(self, img_fn, device):
        img = PIL.Image.open(img_fn)
        img = self.transform(img).to(device)

        return img

    def read_disp(self):
        pass