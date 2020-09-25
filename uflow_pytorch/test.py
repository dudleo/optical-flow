import unittest

import torch
from uflow-pytorch.network import UFlow

from .transformations import rgb_to_grayscale, \
                             neighbors_to_channels, \
                             census_transform, \
                             census_loss

from torchvision.transforms import ToPILImage

from .datasets import KittiDataset

class Tests(unittest.TestCase):

    def test_network(self):
        uflow = UFlow()
        print(uflow.dev)
        uflow.cuda()
        self.assertTrue(uflow)

        x_in = torch.rand((12, 2*3, 512, 512))
        x_out = uflow(x_in.cuda())

        self.assertTrue(x_out is not None)

    def test_rgb_to_grayscale(self):

        B = 12
        H = 512
        W = 512
        x_in = torch.rand((B, 3, H, W))

        x_out = rgb_to_grayscale(x_in.cuda())

        self.assertTrue(x_out.size() == torch.Size([B, 1, H, W]))

    def test_neighbors_to_channels(self):
        B = 12
        C = 3
        H = 512
        W = 512
        x_in = torch.rand((B, C, H, W)).cuda()

        P = 3

        x_out = neighbors_to_channels(x_in, P)

        for i in range(C):
            self.assertTrue(torch.equal(x_in[:, i], x_out[:, 4+i*9]))

    def test_census_transform(self):
        B = 12
        C = 1
        H = 512
        W = 512

        x_in = torch.rand((B, C, H, W)).cuda()

        x_out = census_transform(x_in, patch_size=7)

        self.assertTrue(x_out.size() == torch.Size([B, 49, H, W]))

    def test_census_loss(self):
        B = 12
        C = 3
        H = 512
        W = 512

        x1_in = torch.rand((B, C, H, W)).cuda()
        x2_in = torch.rand((B, C, H, W)).cuda()

        x_out = census_loss(x1_in, x2_in, patch_size=7)

        self.assertTrue(x_out.size() == torch.Size([]))

    def test_kitti_dataset(self):
        kitti_dataset = KittiDataset('datasets/KITTI_flow_multiview/training/image_2')

        kitti_dataloader = torch.utils.data.DataLoader(kitti_dataset, batch_size=10, shuffle=True)

        for i, imgpair in enumerate(kitti_dataloader):
            print(i, imgpair.size())
            if i == 3:
                break

    def test_warping(self):
        kitti_dataset = KittiDataset(imgs_dir='..\\datasets\\KITTI_flow\\training\\image_2',
                                     flow_dir='..\\datasets\\KITTI_flow\\training\\flow_noc', preload=False)

        kitti_dataloader = torch.utils.data.DataLoader(kitti_dataset, batch_size=1, shuffle=False)

        pil_transform = ToPILImage()
        for i, (imgpair, gt_flow_uv, gt_flow_valid) in enumerate(kitti_dataloader):
            img1 = imgpair[0, :3, :, :]
            img2 = imgpair[0, 3:, :, :]

            pil_transform(img1).show()
            input('press key to continue')

if __name__=='__main__':
    unittest.main()

