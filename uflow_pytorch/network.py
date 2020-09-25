
import torch
import torch.nn as nn
import numpy as np

class UFlow(nn.Module):

    def __init__(self):
        super(UFlow, self).__init__()

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dtype = torch.float32
        self.num_levels = 6
        self.output_level = 2
        self.leaky_relu_negative_slope = 0.1

        self.module_leaky_relu = nn.LeakyReLU(negative_slope = self.leaky_relu_negative_slope)

        self.cost_volume_maximum_displacement = 4
        self.cost_volume_channels = (2 * self.cost_volume_maximum_displacement + 1) ** 2
        self.context_channels = 32

        ##  define modules for each level to create feature pyramid ##
        self.level_features_channels = [3, 32, 32, 32, 32, 32]
        self.modules_features = nn.ModuleList([])

        for module_id in range(self.num_levels-1):
            in_channels = self.level_features_channels[module_id]
            out_channels = self.level_features_channels[module_id+1]

            module_features = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope = self.leaky_relu_negative_slope),
                nn.Conv2d(out_channels,
                          out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope = self.leaky_relu_negative_slope),
                nn.Conv2d(out_channels,
                          out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope = self.leaky_relu_negative_slope)
            )

            self.modules_features.append(module_features.to(self.dev))

        ##  define modules for each level to estimate context and optical flow ##

        self.modules_context = nn.ModuleList([])
        self.modules_flow = nn.ModuleList([])
        self.modules_context_upsampling = nn.ModuleList([])

        for module_id in range(self.num_levels-self.output_level):
            cost_volume_channels = self.cost_volume_channels
            features_channels = self.level_features_channels[-(1 + module_id)]
            if module_id > 0:
                flow_channels = 2
                context_channels = self.context_channels
            else:
                flow_channels = 0
                context_channels = 0

            module_in_channels = features_channels + cost_volume_channels + flow_channels + context_channels
            layers_out_channels = [128, 128, 96, 64, 32]
            layers_out_channels_acc = np.cumsum(layers_out_channels)
            module_context_layers_list = nn.ModuleList([])
            for i in range(len(layers_out_channels)):

                if i == 0:
                    layer_in_channels = module_in_channels
                else:
                    layer_in_channels = module_in_channels + layers_out_channels_acc[i-1]

                layer_out_channels = layers_out_channels[i]

                module_context_layer_list = nn.ModuleList([])
                module_context_layer_list.append(nn.Conv2d(layer_in_channels,
                                                        layer_out_channels,
                                                        kernel_size=3, stride=1, padding=1))
                module_context_layer_list.append(nn.LeakyReLU(negative_slope = self.leaky_relu_negative_slope))
                module_context_layer = nn.Sequential(*module_context_layer_list)
                module_context_layers_list.append(module_context_layer.to(self.dev))

                if i == len(layers_out_channels)-1:
                    layer_in_channels = self.context_channels
                    self.modules_context_upsampling.append(nn.ConvTranspose2d(layer_in_channels, self.context_channels,
                                                                 kernel_size=4, stride=2, padding=1).to(self.dev))

                if i == len(layers_out_channels)-1:
                    module_refinement_list = nn.ModuleList([])
                    layers_out_channels = [128, 128, 128, 96, 64, 32]
                    layers_dilations = [1, 2, 4, 8, 16, 1]
                    for j in range(len(layers_out_channels)):
                        if j == 0:
                            layer_in_channels = 2 + self.context_channels
                        else:
                            layer_in_channels = layers_out_channels[j - 1]
                        # else: we inherit the layer_in_channels from the l
                        layer_out_channels = layers_out_channels[j]

                        module_refinement_list.append(nn.Conv2d(layer_in_channels, layer_out_channels,
                                                                dilation=layers_dilations[j],
                                                                kernel_size=3, stride=1, padding=layers_dilations[j]))
                        module_refinement_list.append(nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope))

                    module_refinement_list.append(nn.Conv2d(layers_out_channels[-1], 2,
                                                            kernel_size=3, stride=1, padding=1))
                    self.module_refinement = nn.Sequential(*module_refinement_list)

            ##module_context = nn.Sequential(*module_context_list)
            module_context_layers = module_context_layers_list
            self.modules_context.append(module_context_layers)

            layer_in_channels = self.context_channels
            self.modules_flow.append(nn.Conv2d(layer_in_channels, 2,
                                                     kernel_size=3, stride=1, padding=1).to(self.dev))




        #if self.dev == 'cuda':
        #    self.cuda()
        #else:
        #    self.cpu()

    def forward(self, x):
        # x: B x 2*3 x H x W

        im1 = x[:, :3, :, :]
        im2 = x[:, 3:, :, :]

        im1_feature_pyramid = [im1]
        im2_feature_pyramid = [im2]

        for module_id in range(self.num_levels-1):
            im1_feature_pyramid.append(self.modules_features[module_id](im1_feature_pyramid[-1]))
            im2_feature_pyramid.append(self.modules_features[module_id](im2_feature_pyramid[-1]))

        flow_up = None
        context_up = None
        for module_id in range(self.num_levels-self.output_level):

            im1_features = im1_feature_pyramid[-(1+module_id)]

            if flow_up is None:
                im2_features_warped = im2_feature_pyramid[-(1+module_id)]
            else:
                im2_features_warped = self.warp(im2_feature_pyramid[-(1+module_id)], flow_up)

            cost_volume = self.compute_cost_volume(im1_features, im2_features_warped,
                                                   self.cost_volume_maximum_displacement)

            if flow_up is not None and context_up is not None:
                x = torch.cat((cost_volume, im1_features, flow_up, context_up), dim=1)
            else:
                x = torch.cat((cost_volume, im1_features), dim=1)

            for module in self.modules_context[module_id][:-1]:
                x = torch.cat((x, module(x)), dim=1)

            context = self.modules_context[module_id][-1](x)

            flow = self.modules_flow[module_id](context)

            ## TODO: if training add dropout context, flow
            #if self.training:
            #context *= maybe_dropout
            #flow *= maybe_dropout

            if flow_up is not None:
                flow = flow_up + flow
            else:
                pass

            context_up = self.modules_context_upsampling[module_id](context)

            '''
            import torch
            import torch.nn as nn
            x1 = torch.rand((12, 3, 9, 16))
            x1 = torch.diag(torch.ones(3)).unsqueeze(0).unsqueeze(0)
            x1_up = nn.functional.interpolate(x1, scale_factor=2, mode='bilinear') * 2.0
            '''

            flow_up = nn.functional.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False) * 2.0

        x = torch.cat((flow, context), dim=1)
        flow_delta = self.module_refinement(x)

        ## TODO: if training add dropout flow_delta
        #if self.training:
        #flow_delta *= maybe_dropout

        flow_refined = flow + flow_delta

        return flow_refined

    def warp(self, x, flow):

        B, C, H, W = x.size()
        grid_x, grid_y = torch.meshgrid([torch.arange(1.0, W+1.0, dtype=self.dtype, device=self.dev),
                                         torch.arange(1.0, H+1.0, dtype=self.dtype, device=self.dev)])

        grid_x = grid_x/W * 2.0 - 1.0
        grid_y = grid_y/H * 2.0 - 1.0
        grid = torch.stack((grid_x, grid_y), dim=0)

        flow_global = flow + grid

        # TODO add mask for boundary
        # input: (B, C, Hin​, Win​) and grid: (N, Hout​, Wout​, 2)
        return nn.functional.grid_sample(input=x, grid=flow_global.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)

    def compute_cost_volume(self, x1, x2, max_displacement):
        # x1 : B x C x H x W
        # x2 : B x C x H x W

        '''
        import torch
        import torch.nn as nn
        x1 = torch.rand((12, 3, 9, 16))
        x2 = torch.rand((12, 3, 9, 16))
        max_displacement = 4
        cost_volume = torch.zeros((B, cost_volume_channel_dim, H, W), dtype=torch.float32, device='cpu')
        '''

        x1 = self.normalize_features(x1)
        x2 = self.normalize_features(x2)

        B, C, H, W = x1.size()

        padding_module = nn.ConstantPad2d(max_displacement, 0.0)
        x2_pad = padding_module(x2)

        num_shifts = 2 * max_displacement + 1

        cost_volume_channel_dim = num_shifts**2
        cost_volume = torch.zeros((B, cost_volume_channel_dim, H, W), dtype=self.dtype, device=self.dev)

        for i in range(num_shifts):
            for j in range(num_shifts):
                cost_volume_single_layer = torch.mean(x1*x2_pad[:, :, j:j+H, i:i+W], dim=1)
                cost_volume[:, i*num_shifts + j, :, :] = cost_volume_single_layer


        cost_volume = self.module_leaky_relu(cost_volume)

        return cost_volume

    def normalize_features(self, x1):
        # over channel and spatial dimensions
        # x1 : B x C x H x W

        '''
        x1 = torch.rand((12, 3, 9, 16))
        '''

        var, mean = torch.var_mean(x1, dim=(1, 2, 3), keepdim=True)

        # 1e-16 for robustness of division by std, this can be found in uflow implementation
        std = torch.sqrt(var + 1e-16)
        x1_normalized = (x1-mean) / std

        return x1_normalized


'''
    def pwc_warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()



        if x.is_cuda:
            grid = grid.cuda()

        # not sure why they are using this variable here, might be important
        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask
'''