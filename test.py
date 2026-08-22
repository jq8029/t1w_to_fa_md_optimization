import os
import torch
import numpy as np
import torch.nn as nn
import config as cf
import putils as ut
from modules import TransUNet
from modules import UNet3D
from modules import CNN3D
torch.cuda.set_per_process_memory_fraction(0.90)


def test(source_dir, model, block_size=64, two_head=True):
    count = 0
    pad_sz = 1

    fa_all_mae = []
    fa_all_rmse = []
    fa_all_r = []
    fa_all_ssim = []

    md_all_mae = []
    md_all_rmse = []
    md_all_r = []
    md_all_ssim = []

    for subject_dir in sorted(d for d in os.listdir(source_dir)
                        if os.path.isdir(os.path.join(source_dir, d))):
        count = count + 1
        print(f'Processing subject {count} : {subject_dir}')

        t1w_img, t1w_nii  = ut.load_image(source_dir, subject_dir, cf.T1W_FILE_NAME)
        fa_img, _   = ut.load_image(source_dir, subject_dir, cf.FA_FILE_NAME)
        md_img, _   = ut.load_image(source_dir, subject_dir, cf.MD_FILE_NAME, rescaling=True) 
        mask_img, _ = ut.load_image(source_dir, subject_dir, cf.MASK_FILE_NAME)
        block_index = ut.get_block_index(mask_img, size=block_size)

        t1w_img    = ut.zscore_normalize(t1w_img, mask_img)
        t1w_blocks = ut.get_blocks(t1w_img, block_index)
        fa_blocks  = ut.get_blocks(fa_img, block_index)
        md_blocks  = ut.get_blocks(md_img, block_index)

        net_in = torch.from_numpy(t1w_blocks)

        fa_gt = torch.from_numpy(fa_blocks).contiguous()
        md_gt = torch.from_numpy(md_blocks).contiguous()
        fa_gt = fa_gt.to(device=cf.DEVICE, dtype=torch.float)
        md_gt = md_gt.to(device=cf.DEVICE, dtype=torch.float)
        
        with torch.no_grad():
            crit_test = nn.L1Loss()
            if (two_head):
                fa_pred, md_pred = model(net_in.to(device=cf.DEVICE,dtype=torch.float))
                y_pred = torch.cat([fa_pred, md_pred], dim=1).contiguous()
            else:
                y_pred = model(net_in.to(device=cf.DEVICE,dtype=torch.float))

            y_pred = y_pred.cpu().numpy()
        
        y_pred_wb = ut.assemble_blocks(y_pred, block_index, mask_img)

        fa_pred = y_pred_wb[0,:,:,:]
        md_pred = y_pred_wb[1,:,:,:]
        fa_gt = fa_img
        md_gt = md_img
        mask = mask_img > 0

        # FA metrics
        ssim = ut.compute_ssim(fa_pred, fa_gt, mask)
        mae, rmse, r = ut.compute_metrics(fa_pred, fa_gt, mask)
        fa_all_mae.append(mae)
        fa_all_rmse.append(rmse)
        fa_all_r.append(r)
        fa_all_ssim.append(ssim)
        print(f" FA -  MAE: {mae:.6f}  RMSE: {rmse:.6f}  r: {r:.6f}  SSIM: {ssim:.6f}")

        # MD metrics
        ssim = ut.compute_ssim(md_pred, md_gt, mask)
        mae, rmse, r = ut.compute_metrics(md_pred, md_gt, mask)
        md_all_mae.append(mae)
        md_all_rmse.append(rmse)
        md_all_r.append(r)  
        md_all_ssim.append(ssim)
        print(f" MD -  MAE: {mae:.6f}  RMSE: {rmse:.6f}  r: {r:.6f}  SSIM: {ssim:.6f}")

        # Save predict images if need
        fa_path = os.path.join(source_dir, subject_dir, cf.PRED_FA_NAME)
        ut.save_image(fa_pred, fa_path, t1w_nii)
        md_path = os.path.join(source_dir, subject_dir, cf.PRED_MD_NAME)
        ut.save_image(md_pred, md_path, t1w_nii)

    # ===== FINAL =====
    print("\n===== FINAL RESULTS =====")

    print(f"FA:  MAE = {np.mean(fa_all_mae):.6f},  RMSE = {np.mean(fa_all_rmse):.6f},  r = {np.mean(fa_all_r):.6f},  SSIM = {np.mean(fa_all_ssim):.6f}")
    print(f"MD:  MAE = {np.mean(md_all_mae):.6f},  RMSE = {np.mean(md_all_rmse):.6f},  r = {np.mean(md_all_r):.6f},  SSIM = {np.mean(md_all_ssim):.6f}")


# ---- Test ----

if cf.MODEL == "TransUNet":
    model = TransUNet(two_head=cf.TWO_HEAD).to(cf.DEVICE)
    best_model_file = cf.TRANSUNET_BEST_MODEL
elif cf.MODEL == "UNet":
    model = UNet3D().to(cf.DEVICE)
    best_model_file = cf.UNET_BEST_MODEL
else: 
    model = CNN3D().to(cf.DEVICE)
    best_model_file = cf.CNN_BEST_MODEL

state = torch.load(best_model_file, map_location="cpu")
model.load_state_dict(state, strict=False)

ut.set_seed(cf.SEED)
test(cf.TEST_DIR, model, cf.BLOCK_SIZE, two_head=cf.TWO_HEAD)
