import argparse
import subprocess
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.val_utils import AverageMeter, compute_psnr_ssim, compute_psnr_ssim2
from utils.image_io import save_image_tensor
from utils.image_utils import *

torch.set_float32_matmul_precision('medium')
print(torch.cuda.is_available())
print(torch.cuda.device_count())


def test_real_denoise(net, dataset, degrad_id=None, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'urban/ours/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([1]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch,prompt)
            else:
                restored = net(degrad_patch)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(degrad_patch[:, select_bands, :, :], output_path + 'real_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_' + clean_name[0] + '.png')

        result_str = "Degrad_Id=%d: psnr: %.2f, ssim: %.4f" % (degrad_id, psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'w') as f:
            f.write(result_str + '\n')


def test_poisson_denoise(net, dataset, degrad_id=None, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'poisson/' + str(degrad_id) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([0]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch,prompt)
            else:
                restored = net(degrad_patch)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :], output_path + 'poisson_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_' + clean_name[0] + '.png')

        result_str = "Degrad_Id=%d: psnr: %.2f, ssim: %.4f" % (degrad_id, psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'w') as f:
            f.write(result_str + '\n')


def test_gaussian_denoise(net, dataset, rank=20, sigma=30, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'gaussian_denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([0]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                # restored = net(degrad_patch)
                restored = net(degrad_patch)

            # clean_patch = torch.clamp(clean_patch, 0, 1)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :],
                              output_path + 'gaussian_noisy_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_' + clean_name[0] + '.png')

        result_str = "Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def test_gaussian_inid_denoise(net, dataset, rank=20, sigma=30, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'gaussian_inid_denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([1]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                restored = net(degrad_patch)
            # clean_patch = torch.clamp(clean_patch, 0, 1)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :],
                              output_path + 'gaussian_noisy_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_' + clean_name[0] + '.png')

        result_str = "Denoise sigma=%s: psnr: %.2f, ssim: %.4f" % (str(sigma), psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def test_stripe_denoise(net, dataset, rank=20, stripe_ratio=0.1, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'destripe/' + str(stripe_ratio) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([1]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                restored = net(degrad_patch)
               

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :], output_path + 'striped_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_' + clean_name[0] + '.png')

        result_str = "Destripe stripe ratio=%s: psnr: %.2f, ssim: %.4f" % (str(stripe_ratio), psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def test_deadline_denoise(net, dataset, rank=20, deadline_ratio=0.1, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'deadline_denoise/' + str(deadline_ratio) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([1]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                restored = net(degrad_patch)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :], output_path + 'deadlined_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_' + clean_name[0] + '.png')

        result_str = "Deadline denoise deadline ratio=%s: psnr: %.2f, ssim: %.4f" % (str(deadline_ratio), psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def test_impulse_denoise(net, dataset, rank=20, impulse_ratio=0.1, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'impulse_denoise/' + str(impulse_ratio) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([1]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                restored = net(degrad_patch)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :], output_path + 'impulse_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :],
                              output_path + 'restored_instructir' + clean_name[0] + '.png')

        result_str = "Impulse denoise impulse ratio=%s: psnr: %.2f, ssim: %.4f" % (str(impulse_ratio), psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def test_gaussian_deblur(net, dataset, rank=20, sigma=1.6, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'gaussian_deblur/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([2]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                # restored = net(degrad_patch)
                restored = net(degrad_patch)
            # clean_patch = torch.clamp(clean_patch, 0, 1)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :],
                              output_path + 'gaussian_blurred_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :],
                              output_path + 'restored_instructir' + clean_name[0] + '.png')

        result_str = "Gaussian deblur sigma=%s: psnr: %.2f, ssim: %.4f" % (str(sigma), psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def test_motion_deblur(net, dataset, rank=20, motion_radius=5, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'motion_deblur/' + str(motion_radius) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([0]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                restored = net(degrad_patch)

            clean_patch = torch.clamp(clean_patch, 0, 1)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :],
                              output_path + 'motion_blurred_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_' + clean_name[0] + '.png')

        result_str = "Motion deblur motion radius=%s: psnr: %.2f, ssim: %.4f" % (str(motion_radius), psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def test_super_resolution(net, dataset, rank=20, downsample_factor=4, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'super_resolution/' + str(downsample_factor) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([3]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                # restored = net(degrad_patch)
                restored = net(degrad_patch)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :],
                              output_path + 'downsampled_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_' + clean_name[0] + '.png')

        result_str = "Super resolution downsample factor=%d: psnr: %.2f, ssim: %.4f" % (downsample_factor, psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def test_inpaint(net, dataset, rank=20, mask_ratio=0.1, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'inpaint/' + str(mask_ratio) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch, mask) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([4]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                # restored = net(degrad_patch)
                restored = net(degrad_patch)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :], output_path + 'masked_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_' + clean_name[0] + '.png')

        result_str = "Inpaint mask ratio=%f: psnr: %.2f, ssim: %.4f" % (mask_ratio, psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def test_dehaze(net, dataset, rank=80, haze_omega=0.5, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'dehaze/real' + str(haze_omega) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([5]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                # restored = net(degrad_patch)
                restored = net(degrad_patch)
            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :], output_path + 'hazed_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_hair' + clean_name[0] + '.png')

        result_str = "Dehaze haze omega=%s: psnr: %.2f, ssim: %.4f" % (str(haze_omega), psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def test_bandmis(net, dataset, bandmis_ratio=0.3, select_bands=None, use_conv3d=False, device=None, testopt=None, need_class=False):
    output_path = testopt.output_path + 'bandmis/' + str(bandmis_ratio) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    uncertainty_residual_records = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([5]).to(device)  # natural_scene
            # prompt = torch.tensor([6]).to(device)#remote_sensing

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            if need_class:
                restored = net(degrad_patch, prompt)
            else:
                restored = net(degrad_patch)
            # clean_patch = torch.clamp(clean_patch, 0, 1)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim2(restored, clean_patch,
                                                         degrad_patch)  # Only record the PSNR and SSIM of the complementary band
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:, select_bands, :, :], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:, select_bands, :, :], output_path + 'bandmis_' + clean_name[0] + '.png')
            save_image_tensor(restored[:, select_bands, :, :], output_path + 'restored_' + clean_name[0] + '.png')

        result_str = "Bandmiss ratio=%f: psnr: %.2f, ssim: %.4f" % (bandmis_ratio, psnr.avg, ssim.avg)
        print(result_str)
        result_file = os.path.join(output_path, '1_result.txt')
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(result_str + '\n')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

