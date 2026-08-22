import os
import torch
import random
import numpy as np
import config as cf
import nibabel as nib
import torch.nn.functional as F
from dataset import Dataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_block(x, size):
    min, max = np.min(x), np.max(x)
    len = max - min + 1
    num = int(np.ceil(len / size)) + 1
    start = min
    end = max - size + 1
    block = np.round(
        np.linspace(start, end, num)
    ).astype(int)

    return block

def get_block_index(mask, size=64):
    """
    Input:  [D, H, W]
    Output: [N, C, D, H, W]
    """
    d, h, w = np.nonzero(mask)

    d_block = get_block(d, size)
    h_block = get_block(h, size)
    w_block = get_block(w, size)

    ind_block = []
    for d in d_block:
        for h in h_block:
            for w in w_block:
                ind_block.append([
                    d, d + size - 1,
                    h, h + size - 1,
                    w, w + size - 1
                ])

    return np.array(ind_block, dtype=int)


def get_blocks(image, index):
    d = index[0, 1] - index[0, 0] + 1
    h = index[0, 3] - index[0, 2] + 1
    w = index[0, 5] - index[0, 4] + 1

    n = index.shape[0]
    blocks = np.zeros(
        (n, 1, d, h, w),
        dtype=image.dtype
    )

    for i in range(n):
        inds_this = index[i, :]

        blocks[i, 0, :, :, :] = image[
            inds_this[0]:inds_this[1] + 1,
            inds_this[2]:inds_this[3] + 1,
            inds_this[4]:inds_this[5] + 1
        ]

    # output: (n, 1, d, h, w)
    return blocks


def assemble_blocks(blocks, index, mask):
    n, c, _, _, _ = blocks.shape
    d, h, w = mask.shape

    image = np.zeros(
        (c, d, h, w),
        dtype=blocks.dtype
    )

    for i in range(n):
        d0, d1, h0, h1, w0, w1 = index[i]

        image[
            :,
            d0:d1 + 1,
            h0:h1 + 1,
            w0:w1 + 1
        ] = blocks[i]

    return image

def zscore_normalize(x, mask):
    """
    x:      (d, h, w)
    mask:   (d, h, w)
    return: (d, h, w)
    """
    mask = mask.astype(np.float32)

    num = mask.sum(axis=(0, 1, 2), keepdims=True)
    num = np.clip(num, a_min=1.0, a_max=None)

    mean = (x * mask).sum(axis=(0, 1, 2), keepdims=True) / num
    var = ((x - mean) ** 2 * mask).sum(axis=(0, 1, 2), keepdims=True) / num
    std = np.sqrt(var)
    std = np.clip(std, a_min=1e-6, a_max=None)

    x_norm = (x - mean) / std
    x_norm = x_norm * mask

    return x_norm


def load_image(root_dir, subject_dir, file_name, rescaling=False):
    full_path = os.path.join(root_dir,subject_dir, file_name)
    nii_file = nib.load(full_path)

    img_data = nii_file.get_fdata(dtype=np.float32)  # (x,y,z)
    if (rescaling):
        img_data = img_data / 3 * 1e3
#    img_data = img_data[np.newaxis, ...]   # (1,x,y,z)

    return img_data, nii_file


def load_data(source_dir, block_size=64):
    all_data_t1w = []
    all_data_fa = []
    all_data_md = []

    # loop all dirs under root dir
    for subject_dir in sorted(d for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d))):
        t1w_img, _  = load_image(source_dir, subject_dir, cf.T1W_FILE_NAME)
        fa_img, _   = load_image(source_dir, subject_dir, cf.FA_FILE_NAME)
        md_img, _   = load_image(source_dir, subject_dir, cf.MD_FILE_NAME, rescaling=True) 
        mask_img, _ = load_image(source_dir, subject_dir, cf.MASK_FILE_NAME)
        block_index = get_block_index(mask_img, size=block_size)

        t1w_img    = zscore_normalize(t1w_img, mask_img)
        t1w_blocks = get_blocks(t1w_img, block_index)
        fa_blocks  = get_blocks(fa_img, block_index)
        md_blocks  = get_blocks(md_img, block_index)
 
        all_data_t1w.append(t1w_blocks)
        all_data_fa.append(fa_blocks)
        all_data_md.append(md_blocks)

    all_data_t1w = np.concatenate(all_data_t1w, axis=0)
    all_data_fa  = np.concatenate(all_data_fa,  axis=0)
    all_data_md  = np.concatenate(all_data_md,  axis=0)

    return all_data_t1w, all_data_fa, all_data_md

def compute_ssim_mean(pred, true):
    ssim_vals = []
    for i in range(true.shape[2]):  # Z-dim
        dr = true[:, :, i].max() - true[:, :, i].min()

        if dr == 0:
            dr = 1.0

        ssim_val = ssim_fn(pred[:, :, i], true[:, :, i], data_range=dr)
        ssim_vals.append(ssim_val)

    return np.mean(ssim_vals)


def make_loaders(
    input_train: np.ndarray, output_train: np.ndarray,
    input_val: np.ndarray = None, output_val: np.ndarray = None,
    batch_size: int = 1, num_workers: int = 0, augment: bool = True
):
    train_ds = Dataset(input_train, output_train, augment=augment)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = None
    if input_val is not None and output_val is not None:
        val_ds = Dataset(input_val, output_val, augment=False)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loader


def fa_ssim_loss(x, y):
    C1=0.01**2
    C2=0.03**2
    x = torch.clamp(x, 0, 1)
    y = torch.clamp(y, 0, 1)

    mu_x = F.avg_pool3d(x, 3, 1, 1)
    mu_y = F.avg_pool3d(y, 3, 1, 1)

    sigma_x = F.avg_pool3d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool3d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool3d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    )

    return torch.clamp(ssim.mean(), 0, 1)

def compute_loss(pred_fa, gt_fa, pred_md, gt_md, epoch, fa_ssim=0.0):
    md_v = F.l1_loss(pred_md, gt_md)
    fa_v = F.l1_loss(pred_fa, gt_fa)

    if (fa_ssim > 0):
        ssim_v = fa_ssim_loss(pred_fa, gt_fa)
        loss_fa = fa_v + fa_ssim * (1 - ssim_v)
    else:
        loss_fa = fa_v

    loss_md = md_v

    loss = loss_fa + loss_md

    return loss


def compute_ssim(pred, gt, mask):
    coords = np.where(mask > 0)

    xmin, xmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    zmin, zmax = coords[2].min(), coords[2].max()

    pred_crop = pred[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
    gt_crop   = gt  [xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
    mask_crop = mask[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]

    ssim_list = []

    for i in range(pred_crop.shape[2]):
        gt_2d = gt_crop[:, :, i]
        pred_2d = pred_crop[:, :, i]
        mask_2d = mask_crop[:, :, i]

        if mask_2d.sum() == 0:
            continue

        val = ssim_fn(
            gt_2d,
            pred_2d,
            data_range=1.0
        )

        ssim_list.append(val)

    if len(ssim_list) == 0:
        return np.nan

    return np.mean(ssim_list)


def compute_metrics(pred, gt, mask=None):
    if mask is not None:
        pred = pred[mask > 0]
        gt = gt[mask > 0]
    else:
        pred = pred.flatten()
        gt = gt.flatten()

    # MAE
    mae = np.mean(np.abs(pred - gt))

    # RMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    # Pearson r
    r = np.corrcoef(pred, gt)[0, 1]

    return mae, rmse, r


def save_image(img, img_path, ref_img):
    nifti_img = nib.Nifti1Image(
        img,
        affine=ref_img.affine,
        header=ref_img.header
    )

    nib.save(nifti_img, img_path)
