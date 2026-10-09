import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import *

from net.S2UM import S2UM

from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.val_utils import compute_psnr_ssim
import argparse

torch.set_float32_matmul_precision('medium')
print(torch.cuda.is_available())
print(torch.cuda.device_count())

torch.set_num_threads(1)


def cal_sam(Itrue, Ifake):
    esp = 1e-6
    # element-wise product
    # torch.sum(dim=1) 沿通道求和
    # [B C H W] * [B C H W] --> [B C H W]  Itrue*Ifake
    # [B 1 H W] InnerPro(keepdim)

    InnerPro = torch.sum(Itrue * Ifake, 1, keepdim=True)
    # print('InnerPro')
    # print(InnerPro.shape)
    # 沿通道求范数
    # len1  len2  [B 1 H W] (keepdim)
    len1 = torch.norm(Itrue, p=2, dim=1, keepdim=True)
    len2 = torch.norm(Ifake, p=2, dim=1, keepdim=True)
    # print('len1')
    # print(len1.shape)

    divisor = len1 * len2
    mask = torch.eq(divisor, 0)
    divisor = divisor + (mask.float()) * esp
    cosA = torch.sum(InnerPro / divisor, 1).clamp(-1 + esp, 1 - esp)
    # print(cosA.shape)
    sam = torch.acos(cosA)
    sam = torch.mean(sam) / np.pi
    return sam


class PromptIRModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.loss_fn = nn.L1Loss()
        channel = 31 if self.args.data_type == "natural_scene" else 100

        self.net = S2UM(inp_channels=channel, out_channels=channel,dim=96)

    def forward(self, x):
        return self.net(x)[0]

    def training_step(self, batch, batch_idx):
        [clean_name, de_type], degrad_patch, clean_patch, prompt = batch

        restored_LQ, spa_var_1, spe_var_1, restored_LQ_2, spa_var_2, spe_var_2, loss_balance = self.net(degrad_patch)
        spa_var_1_new = torch.exp(-spa_var_1)
        spe_var_1_new = torch.exp(-spe_var_1)
        spa_var_2_new = torch.exp(-spa_var_2)
        spe_var_2_new = torch.exp(-spe_var_2)

        loss_uncertainty = (0.5 * (self.loss_fn(restored_LQ * spa_var_1_new * spe_var_1_new, clean_patch * spa_var_1_new * spe_var_1_new) + 0.1 * torch.mean(
            spa_var_1) + 0.1 * torch.mean(spe_var_1)) +
                            0.5 * (self.loss_fn(restored_LQ_2 * spa_var_2_new * spe_var_2_new,F.interpolate(clean_patch, scale_factor=0.5,mode='bilinear') * spa_var_2_new * spe_var_2_new) + 0.1 * torch.mean(
            spa_var_2) + 0.1 * torch.mean(spe_var_2)))

        loss = (loss_uncertainty + self.loss_fn(restored_LQ, clean_patch) +
                0.5 * self.loss_fn(restored_LQ_2, F.interpolate(clean_patch, scale_factor=0.5,mode='bilinear')) + 0.01 * loss_balance)


        self.log('uncertainty_loss', loss_uncertainty)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=int(min(0.1 * self.args.epochs, 30)),  # 0.1 * self.args.epochs # 30
            max_epochs=self.args.epochs,
            eta_min=1e-6
        )

        # return optimizer
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

def set_seed(seed):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--epochs', type=int, default=300 , help='maximum number of epochs to train the total model.')
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size to use per GPU")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate of encoder.')
    parser.add_argument('--init', type=str, default='xu',
                        help='which init scheme to choose.', choices=['kn', 'ku', 'xn', 'xu'])

    parser.add_argument('--mode', type=int, default=0, help='Degraded Mode.')
    parser.add_argument('--natural_scene_single_de_type', nargs='+',
                        default=['gaussianN', 'complexN', 'blur', 'sr', 'inpaint', 'bandmiss'],
                        help='which type of single degradation is training and testing for.')
    parser.add_argument('--remote_sensing_single_de_type', nargs='+',
                        default=['gaussianN', 'complexN', 'blur', 'sr', 'inpaint', 'haze', 'bandmiss'],
                        # 'gaussianN', 'complexN', 'blur', 'sr', 'inpaint', 'haze', 'bandmiss'
                        help='which type of single degradation is training and testing for.')

    parser.add_argument('--patch_size', type=int, default=64, help='patchsize of input.')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')

    parser.add_argument('--data_type', type=str, default="remote_sensing",
                        help="Types of data used for training.")
    parser.add_argument('--classifier', type=bool, default=False, help="")

    parser.add_argument('--db_path', type=str,
                        default='Remote_sensing_minmax_patch_64.db/',
                        help='where clean HSIs of remote_sensing saves.')
    parser.add_argument('--classifier_path', type=str, default='',
                        help='')

    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default=None, help='checkpoint save path')

    parser.add_argument("--ckpt_dir", type=str, default='',
                        help="Name of the Directory where the checkpoint is to be saved")
    parser.add_argument("--num_gpus", type=list, default=[1], help="Number of GPUs to use for training")
    parser.add_argument("--repeat", type=int, default=1, help="")

    opt = parser.parse_args()

    model = PromptIRModel(opt)

    model_name = model.net.__class__.__name__

    print("Options")
    print(opt)
    set_seed(opt.seed)
    if opt.data_type == "natural_scene":
        log_name = "Natural_scene"
    elif opt.data_type == "remote_sensing":
        log_name = "Remote_sensing"
    else:
        log_name = "Other"
    logger = TensorBoardLogger(save_dir="./logs/" + log_name, name=model_name)

    # train dataloader
    train_data = LMDBDataset(opt)
    trainset = ImageTransformDataset(train_data, opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, drop_last=True, pin_memory=True,
                             num_workers=opt.num_workers, persistent_workers=True)


    if opt.data_type == "natural_scene":
        classifier_ckpt_path = "./ckpt/Natural_scene"
    else:
        classifier_ckpt_path = "./ckpt/Remote_sensing"
    classifier_ckpt_path = os.path.join(classifier_ckpt_path, model_name)
    if not os.path.exists(classifier_ckpt_path):
        os.makedirs(classifier_ckpt_path)
    checkpoint_callback = ModelCheckpoint(dirpath=classifier_ckpt_path, every_n_epochs=1, save_top_k=-1)

    if opt.ckpt_path is not None:
        checkpoint = torch.load(opt.ckpt_path, map_location=torch.device(f"cuda:{opt.num_gpus[0]}"))
        state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()

        filtered_dict = {k: v for k, v in state_dict.items() if
                         k in model_state_dict and model_state_dict[k].shape == v.shape}
        model_state_dict.update(filtered_dict)
        model.load_state_dict(model_state_dict, strict=False)

    trainer = pl.Trainer(precision="16-mixed",max_epochs=opt.epochs, devices=opt.num_gpus, accelerator="gpu",
                         logger=logger, callbacks=[checkpoint_callback],
                         check_val_every_n_epoch=1)

    trainer.fit(model=model, train_dataloaders=trainloader)

from test_function import (test_real_denoise, test_poisson_denoise, test_gaussian_denoise, test_gaussian_inid_denoise,
                          test_stripe_denoise,
                          test_deadline_denoise, test_impulse_denoise, test_gaussian_deblur, test_motion_deblur,
                          test_super_resolution,
                          test_inpaint, test_dehaze, test_bandmis)


def test():
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--mode', type=int, default=0,
                        help='Used to select degradation mode.')
    # 直接读取的test文件夹下的mat文件
    parser.add_argument('--test_dir', type=str, default="./testdata/ICVL",
                        help='where clean HSIs of test saves.')
    parser.add_argument('--degrad_id', type=int, default=1, help='')
    parser.add_argument('--gaussian_noise_sigma', type=int, default=70, help='Gaussian Noise intensity')
    parser.add_argument('--gaussian_noise_sigmas', type=list, default=[10, 30, 50, 70],
                        help='Gaussian Noise inid intensity')
    parser.add_argument('--stripe_nosie_ratio', type=str, default=[0.05, 0.15], help='Stripe ratio')
    parser.add_argument('--deadline_nosie_ratio', type=str, default=[0.05, 0.15], help='Deadline ratio')
    parser.add_argument('--impulse_nosie_ratio', type=str, default=[0.1, 0.3, 0.5, 0.7], help='Impulse ratio')
    parser.add_argument('--gaussian_blur_radius', type=str, default=15, help='Gaussian Blur')
    parser.add_argument('--motion_blur_radius', type=tuple, default=(15, 45), help='Motion Blur')
    parser.add_argument('--downsample_factor', type=int, default=8, help='factor')
    parser.add_argument('--mask_ratio', type=str, default=0.9, help='Inpaint Mask Ratio')
    parser.add_argument('--haze_omega', type=str, default=1, help='haze')
    parser.add_argument('--bandmis_ratio', type=str, default=0.3, help='Bandmis Ratio')
    parser.add_argument('--select_bands', type=list, default=[27, 15, 9],
                        help='The bands used to compose the pseudo-color image')
    parser.add_argument('--output_path', type=str,
                        default="./output/",
                        help='output save path')
    parser.add_argument('--ckpt_path', type=str,
                        default="/home/xxx/HSI-AiOR/ckpt/Restormer/Natural_scene/epoch=71-step=300888.ckpt",
                        help='checkpoint save path')
    parser.add_argument('--rank', type=int, default=31,
                        help='When the number of bands is less than 100, it is 20; when it is greater than or equal to 100, it is 80.')
    parser.add_argument("--model_choose", type=str, default='Restormer', help="Choose the model to be trained")
    parser.add_argument('--data_type', type=str, default="natural_scene",
                        help="Types of data used for training.")  # natural_scene, remote_sensing

    testopt = parser.parse_args()

    set_seed(testopt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #

    for data_type in ['remote_sensing']:
        testopt.data_type = data_type
        if data_type == 'natural_scene':
            test_dirs = ["./testdata/ICVL", "./testdata/ARAD"]
            ckpt_path = ""
            model = None
            net = PromptIRModel.load_from_checkpoint(ckpt_path, args=testopt, model=model, map_location=device).to(device)

        else:
            test_dirs = [
                "./testdata/Berlin",
                "./testdata/Chikusei",
                "./testdata/Eagle",
                "./testdata/Houston",
                "./testdata/PaviaC",
                "./testdata/PaviaU",
                "./testdata/WDC",
                "./testdata/Xiongan"
            ]
            ckpt_path = ".ckpt"
            net = PromptIRModel.load_from_checkpoint(ckpt_path, args=testopt, map_location=device, strict=True).to(device)
        model_name = net.net.__class__.__name__
        net.eval()
        for test_dir in test_dirs:
            testopt.output_path = "./output/"
            testopt.test_dir = test_dir
            data_name = testopt.test_dir.split('/')[-1]
            testopt.output_path = testopt.output_path + model_name + '/' + data_name + "/"
            if data_name in ['ICVL', 'ARAD']:
                testopt.select_bands = [27, 15, 9]
                gaussian_blur_radius_rand = [9, 15, 21]
            else:
                testopt.select_bands = [20, 80, 50]
                gaussian_blur_radius_rand = [7, 11, 15]

            for mode in range(12):
                testopt.mode = mode
                if testopt.mode == 0:
                    for gaussian_noise_sigma in [30, 50, 70]:
                        testopt.gaussian_noise_sigma = gaussian_noise_sigma
                        gaussian_denoise_testset = Gaussian_Denoise_Dataset(testopt)
                        print('Start gaussian denoise testing sigma={}'.format(testopt.gaussian_noise_sigma))
                        test_gaussian_denoise(net, gaussian_denoise_testset, rank=testopt.rank,
                                              sigma=testopt.gaussian_noise_sigma,
                                              select_bands=testopt.select_bands, device=device, testopt=testopt)

                if testopt.mode == 1:
                    gaussian_denoise_testset = Gaussian_Denoise_inid_Dataset(testopt)
                    print('Start inid gaussian denoise testing sigma={}'.format(testopt.gaussian_noise_sigmas))
                    test_gaussian_inid_denoise(net, gaussian_denoise_testset, rank=testopt.rank,
                                               sigma=testopt.gaussian_noise_sigmas, select_bands=testopt.select_bands,
                                               device=device, testopt=testopt)

                elif testopt.mode == 2:
                    print('Start destripe testing stripe ratio={}'.format(testopt.stripe_nosie_ratio))
                    destripe_denoise_testset = Stripe_Denoise_Dataset(testopt)
                    test_stripe_denoise(net, destripe_denoise_testset, rank=testopt.rank,
                                        stripe_ratio=testopt.stripe_nosie_ratio,
                                        select_bands=testopt.select_bands, device=device, testopt=testopt)

                elif testopt.mode == 3:
                    print('Start deadline denoise testing deadline ratio={}'.format(testopt.deadline_nosie_ratio))
                    deadline_denoise_testset = Deadline_Denoise_Dataset(testopt)
                    test_deadline_denoise(net, deadline_denoise_testset, rank=testopt.rank,
                                          deadline_ratio=testopt.deadline_nosie_ratio,
                                          select_bands=testopt.select_bands, device=device, testopt=testopt)

                elif testopt.mode == 4:
                    print('Start impulse denoise testing impulse ratio={}'.format(testopt.impulse_nosie_ratio))
                    impulse_denoise_testset = Impulse_Denoise_Dataset(testopt)
                    test_impulse_denoise(net, impulse_denoise_testset, rank=testopt.rank,
                                         impulse_ratio=testopt.impulse_nosie_ratio,
                                         select_bands=testopt.select_bands, device=device, testopt=testopt)

                elif testopt.mode == 5:
                    for gaussian_blur_radius in gaussian_blur_radius_rand:
                        testopt.gaussian_blur_radius = gaussian_blur_radius
                        print('Start gaussian deblur testing sigma={}'.format(testopt.gaussian_blur_radius))
                        gaussian_deblur_testset = Gaussian_Deblur_Dataset(testopt)
                        test_gaussian_deblur(net, gaussian_deblur_testset, rank=testopt.rank,
                                             sigma=testopt.gaussian_blur_radius,
                                             select_bands=testopt.select_bands, device=device, testopt=testopt)

                elif testopt.mode == 6:
                    print('Start Motion deblur testing motion radius={}'.format(testopt.motion_blur_radius))
                    motion_deblur_testset = Motion_Deblur_Dataset(testopt)
                    test_motion_deblur(net, motion_deblur_testset, rank=testopt.rank,
                                       motion_radius=testopt.motion_blur_radius,
                                       select_bands=testopt.select_bands, device=device, testopt=testopt)

                elif testopt.mode == 7:
                    for downsample_factor in [2, 4, 8]:
                        testopt.downsample_factor = downsample_factor
                        print('Start super-resolution testing downsampling factor={}'.format(testopt.downsample_factor))
                        super_resolution_testset = Super_Resolution_Dataset(testopt)
                        test_super_resolution(net, super_resolution_testset, rank=testopt.rank,
                                              downsample_factor=testopt.downsample_factor,
                                              select_bands=testopt.select_bands, device=device, testopt=testopt)

                elif testopt.mode == 8:
                    for mask_ratio in [0.7, 0.8, 0.9]:
                        testopt.mask_ratio = mask_ratio
                        print('Start inpaint testing mask ratio ={}'.format(testopt.mask_ratio))
                        inpaint_testset = Inpaint_Dataset(testopt)
                        test_inpaint(net, inpaint_testset, rank=testopt.rank, mask_ratio=testopt.mask_ratio,
                                     select_bands=testopt.select_bands, device=device, testopt=testopt)

                elif testopt.mode == 9:
                    if data_name in ['ICVL', 'ARAD']:
                        continue
                    else:
                        for haze_omega in [0.5, 0.75, 1]:
                            testopt.haze_omega = haze_omega
                            print('Start dehaze testing haze omega ={}'.format(testopt.haze_omega))
                            dehaze_testset = Dehaze_Dataset(testopt)
                            test_dehaze(net, dehaze_testset, rank=testopt.rank, haze_omega=testopt.haze_omega,
                                        select_bands=testopt.select_bands, device=device, testopt=testopt)

                elif testopt.mode == 10:
                    for bandmis_ratio in [0.1, 0.2, 0.3]:
                        testopt.bandmis_ratio = bandmis_ratio
                        print('Start bandmis ratio ={}'.format(testopt.bandmis_ratio))
                        bandmis_testset = Bandmis_Dataset(testopt)
                        test_bandmis(net, bandmis_testset, bandmis_ratio=testopt.bandmis_ratio,
                                     select_bands=testopt.select_bands, device=device, testopt=testopt)

                elif testopt.mode == 11:
                    print('Start poisson degradation testing {}'.format(testopt.degrad_id))
                    multi_degrad_testset = Poisson_Denoise_Dataset(testopt)
                    test_poisson_denoise(net, multi_degrad_testset, degrad_id=testopt.degrad_id,
                                         select_bands=testopt.select_bands,
                                         device=device, testopt=testopt)

                elif testopt.mode == 12:
                    print('Start real noise degradation testing {}'.format(testopt.degrad_id))
                    multi_degrad_testset = Real_Degrad_Dataset(testopt)
                    test_real_denoise(net, multi_degrad_testset, degrad_id=testopt.degrad_id,
                                      select_bands=testopt.select_bands,
                                      device=device, testopt=testopt)


if __name__ == '__main__':
    # train()
    test()
