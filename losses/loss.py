import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import pytorch_msssim

from losses.loss_utils import weighted_loss
#from basicsr.metrics.lpips.lpips import LPIPS

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@weighted_loss
def log_mse_loss(pred, target):
    return torch.log(F.mse_loss(pred, target, reduction='none'))


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

@weighted_loss
def psnr_loss(pred, target): # NCHW
    mseloss = F.mse_loss(pred, target, reduction='none').mean((1,2,3))
    psnr_val = 10 * torch.log10(1 / mseloss).mean().item()
    return psnr_val

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(PSNRLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * psnr_loss(pred, target, weight) * -1.0 

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)

class PixelLoss(nn.Module):
    """Pixel loss.

    Args:
        loss_weight (float): Loss weight for pixel loss. Default: 1.0.
        criterion (str): Criterion used for pixel loss. Default: 'l1'.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, criterion='l1', reduction='mean'):
        super(PixelLoss, self).__init__()

        self.loss_weight = loss_weight
        self.criterion = criterion
        self.reduction = reduction

        if criterion == 'l1':
            self.loss = L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.loss = MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        loss = self.loss(pred, target, weight, **kwargs)
        return loss, self.loss_weight * loss


class FourierEdgeLoss(nn.Module):
    def __init__(self,
                 crop_size=24,
                 loss_weight=1.0,
                 criterion='l1',
                 reduction='mean', radius_ratio = 0.04):
        super().__init__()

        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}')

        self.loss_weight = loss_weight
        self.crop_size = crop_size

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

        # construir máscara UNA vez

        H = W = self.crop_size
        mask = self._build_highpass_mask(H, W, radius_ratio)
        # buffer = no gradientes, se mueve con .to(), se guarda en state_dict
        self.register_buffer('mask', mask)

    def _build_highpass_mask(self, H, W, radius_ratio):
        cy, cx = H // 2, W // 2

        ys = torch.arange(H) - cy
        xs = torch.arange(W // 2 + 1) - cx

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        dist = torch.sqrt(grid_x ** 2 + grid_y ** 2)

        diag = (H ** 2 + W ** 2) ** 0.5
        r0 = radius_ratio * diag

        sigma = 0.15 * r0
        k = 2.5

        mask = torch.zeros_like(dist)

        low  = dist <= (r0 - k * sigma)
        high = dist >= (r0 + k * sigma)
        mid  = (~low) & (~high)

        mask[high] = 1.0
        d = dist[mid] - r0
        mask[mid] = 1.0 - torch.exp(-(d ** 2) / (2 * sigma ** 2))

        return mask 

    def _extract_high_freq(self, img):
        freq = torch.fft.rfft2(img)
        freq_shifted = torch.fft.fftshift(freq, dim=(-2,))

        freq_shifted = freq_shifted * self.mask[None, None, :, :]

        freq_unshifted = torch.fft.ifftshift(freq_shifted, dim=(-2,))
        high_freq = torch.fft.irfft2(freq_unshifted, s=img.shape[-2:])

        return high_freq

    def forward(self, pred, target, scale_factor = 16):

        target_low = F.interpolate(target, scale_factor=1/scale_factor, mode = 'nearest')
        # pred_edges = self._extract_high_freq(pred)
        target_edges = self._extract_high_freq(target_low)
        loss = self.criterion(pred, target_edges)

        return loss, self.loss_weight * loss


#-----------------------------------------------------------------------------
# define the perceptual loss
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion = 'l1', reduction='mean'):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')


        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')
        
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weight = loss_weight

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return self.weight * loss

#---------------------------------------------------------------
#define the edge loss to enhance the deblurring task
class EdgeLoss(nn.Module):
    def __init__(self, rank, loss_weight=1.0, criterion = 'l2',reduction='mean'):
        super(EdgeLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')


        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')        

        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).to(rank)

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y, scale_factor = 16):
        y_low = F.interpolate(y, scale_factor=1/scale_factor, mode = 'nearest')
        loss = self.criterion(self.laplacian_kernel(x), self.laplacian_kernel(y_low))
        return loss, loss*self.weight

def SSIM_loss(pred_img, real_img, data_range):
    SSIM_loss = pytorch_msssim.ssim(pred_img, real_img, data_range = data_range)
    return SSIM_loss

class SSIM(nn.Module):
    def __init__(self, loss_weight=1.0, data_range = 1.):
        super(SSIM, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * SSIM_loss(pred, target, self.data_range)

class SSIMloss(nn.Module):
    def __init__(self, loss_weight=1.0, data_range = 1.):
        super(SSIMloss, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * (1 - SSIM_loss(pred, target, self.data_range))

class EnhanceLoss(nn.Module):
    '''
    Applies the enhanceLoss. This loss is the l1 loss of the image downsampled at the middle of the
    encoder-decoder plus the l1 of the features of this downsample image given by the vgg19 (a perceptual
    element).
    '''
    def __init__(self, loss_weight=1.0, criterion = 'l1', reduction='mean'):
        super(EnhanceLoss, self).__init__()
        self.loss_weight = loss_weight
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss') 
        
        self.vgg19 = VGGLoss(loss_weight = 0.01,
                              criterion = criterion,
                              reduction = 'mean')
        
    def forward(self, gt, enhanced, scale_factor = 16):
        gt_low_res = F.interpolate(gt, scale_factor=scale_factor, mode = 'nearest')
        loss = self.criterion(gt_low_res, enhanced)
        return loss, self.vgg19(gt_low_res, enhanced) + self.loss_weight * loss

class T_transform(nn.Module):
    def __init__(self, u=5000):
        super(T_transform, self).__init__()
        self.u=u
    def forward(self, img):
        return torch.log(1 + self.u * img) / torch.log(1 + self.u)
    

class L_real(nn.Module):
    def __init__(self, transform=T_transform, l1=L1Loss):
        super().__init__()
        self.transform= T_transform()
        self.L1Loss= L1Loss()
        self.Perceptual = VGGLoss()
    def forward(self, img_r, img_gt):
        return self.L1Loss(self.transform(img_r), self.transform(img_gt)) + self.Perceptual(self.transform(img_r), self.transform(img_gt))

class ClassificationLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(ClassificationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, gt, predicted):
        loss = self.loss(predicted, gt)
        
        return loss, self.loss_weight * loss
    
def calculate_accuracy(preds, labels):
    '''
    Calculate the accuracy of the model.
    '''
    preds = torch.argmax(preds, dim = 1)
    return (preds == labels).float().mean()

class MultiguidanceLoss(nn.Module):
    """
    Multi-guidance loss que calcula la pérdida en múltiples niveles del decoder.
    Las salidas vienen convertidas a RGB mediante side_out convolutions.
    
    Args:
        loss_weight (float): Pesos global de la pérdida.
        criterion (str): Criterio usado ('l1', 'l2', 'charbonnier'). Default: 'l1'.
        reduction (str): Tipo de reducción ('none', 'mean', 'sum'). Default: 'mean'.
        loss_type (str): Tipo de pérdida a aplicar ('enhance', 'edge'). Default: 'edge'.
        rank (int): Device rank para EdgeLoss. Solo necesario si loss_type='edge'.
    """
    def __init__(self, loss_weight=1.0, criterion='l1', reduction='mean', loss_type='edge', cutoff = [0.1,0.1,0.1,0.1], rank=0, crop_size = 384):
        super(MultiguidanceLoss, self).__init__()
        
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                           f'Supported ones are: {_reduction_modes}')
        
        self.loss_weight = loss_weight
        self.loss_type = loss_type.lower()
        crop_sizes = [crop_size//(2 ** i) for i in range(len(cutoff))]

                
        if self.loss_type == 'enhance':
            # Usar EnhanceLoss
            self.criterion = EnhanceLoss(
                loss_weight=self.loss_weight,  # El peso global se aplica después
                criterion=criterion,
                reduction=reduction
            )
            
        elif self.loss_type == 'edge':
            # Usar EdgeLoss
            self.criterion = EdgeLoss(
                rank=rank,
                loss_weight=self.loss_weight,  # El peso global se aplica después
                criterion=criterion,
                reduction=reduction
            )
        elif self.loss_type == 'fourier':
             self.criterion = nn.ModuleList([
                FourierEdgeLoss(
                    crop_size=cs,
                    loss_weight=self.loss_weight,
                    criterion=criterion,
                    reduction=reduction,
                    radius_ratio=cut,
                )
                for cs, cut in zip(crop_sizes, cutoff)
            ])
    
    def forward(self, out_side, target):
        """
        Args:
            out_side (list): Lista de salidas RGB de cada nivel del decoder.
            target (Tensor): Ground truth de shape (N, C, H, W).
        
        Returns:
            Tensor: Pérdida calculada.
        """
        num_levels = len(out_side)
        
        total_loss = 0
        total_weighted = 0
        
        # Calcular las escalas para cada nivel
        
        scales = [2 ** i for i in range(num_levels)]
        
        for i, (side_output, current_scale) in enumerate(zip(out_side, scales)):
                
            if self.loss_type == 'enhance':
                
                level_loss, level_weighted = self.criterion(target, side_output, scale_factor=current_scale)
                
            elif self.loss_type == 'edge':
                
                level_loss, level_weighted = self.criterion(side_output, target, scale_factor=current_scale)

            elif self.loss_type == 'fourier':
                    level_loss, level_weighted = self.criterion[i](side_output, target, scale_factor=current_scale)

            
            total_loss += level_loss
            total_weighted += level_weighted
        
        return total_loss, total_weighted