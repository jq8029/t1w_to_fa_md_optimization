import torch
import time
import putils as ut
import config as cf
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch import amp
from modules import TransUNet
from modules import UNet3D
from modules import CNN3D
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
torch.cuda.set_per_process_memory_fraction(0.90)


def train(model, best_model_file, train_loader, val_loader, fa_ssim=0.0, two_head=True):
   
    scaler = amp.GradScaler()

    # optimizer
    opt = optim.Adam(model.parameters(), lr=3e-5)

    # LR Scheduler
    warmup_epochs = cf.WARMUP_EPOCHS
    total_epochs = cf.EPOCHS
    scheduler = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=0.01, total_iters=warmup_epochs),
            CosineAnnealingLR(opt, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
        ],
        milestones=[warmup_epochs]
    )

    # --- Training config ---
    num_epochs = cf.EPOCHS
    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_time = time.time()
    begin_time = time.time()
    train_losses = []
    val_losses = []
    best_epoch = 0

    for epoch in range(num_epochs):
        # ------- TRAIN -------
        start_time = time.time()
        model.train()
        print('Epoch:', epoch+1)
        running_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(cf.DEVICE), yb.to(cf.DEVICE)
            opt.zero_grad()
    
            # forward: with AMP
            with amp.autocast("cuda"):
                if (two_head) : 
                    pred_fa, pred_md = model(xb)
                else : 
                    pred = model(xb)
                    pred_fa = pred[:, 0:1]
                    pred_md = pred[:, 1:2]

            gt_fa = yb[:, 0:1]
            gt_md = yb[:, 1:2]

            loss = ut.compute_loss(pred_fa, gt_fa, pred_md, gt_md, epoch, fa_ssim=fa_ssim)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running_train_loss += loss.item() * xb.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)

        # ------ VALIDATION --------
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cf.DEVICE), yb.to(cf.DEVICE)
                if (two_head) : 
                    pred_fa, pred_md = model(xb)
                else : 
                    pred = model(xb)
                    pred_fa = pred[:, 0:1]
                    pred_md = pred[:, 1:2]

                gt_fa = yb[:, 0:1]
                gt_md = yb[:, 1:2]
                loss = ut.compute_loss(pred_fa, gt_fa, pred_md, gt_md, epoch)
                running_val_loss += loss.item() * xb.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)

        scheduler.step()
        current_lr = opt.param_groups[0]['lr']
        print(f" Epoch {epoch+1:03d}: Train Loss = {train_loss:.6f},  Val Loss = {val_loss:.6f},  Difference = {(val_loss-best_val_loss):.6f},  LR={current_lr:.6e},  Time = {(time.time()-start_time):.2f},  Total = {(time.time()-begin_time):.2f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ------- CHECKPOINT -------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_file)
            print(f"  Improved val loss. Model saved (best_val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

    print(f"Training complete. Best validation loss = {best_val_loss:.6f}，  Best Epcho = {best_epoch},  Time = {time.time()-start_time}")

    return train_losses, val_losses


# --- Load Data ---

# Load training data
input_train, all_data_fa, all_data_md = ut.load_data(cf.TRAIN_DIR, block_size=cf.BLOCK_SIZE)
output_train = np.concatenate([all_data_fa, all_data_md], axis=1)
print("All training data loaded: ", input_train.shape, output_train.shape)

# Load validation data
input_val, all_data_fa, all_data_md = ut.load_data(cf.VALIDATION_DIR, block_size=cf.BLOCK_SIZE)
output_val = np.concatenate([all_data_fa, all_data_md], axis=1)
print("All validation data loaded: ", input_val.shape, output_val.shape)

# Create train/val loader
train_loader, val_loader = ut.make_loaders(input_train, output_train, input_val, output_val, batch_size=cf.BATCH_SIZE)


# ---- Training ----
if cf.MODEL == "TransUNet":
    model = TransUNet(two_head=cf.TWO_HEAD).to(cf.DEVICE)
    best_model_file = cf.TRANSUNET_BEST_MODEL
elif cf.MODEL == "UNet":
    model = UNet3D().to(cf.DEVICE)
    best_model_file = cf.UNET_BEST_MODEL
else: 
    model = CNN3D().to(cf.DEVICE)
    best_model_file = cf.CNN_BEST_MODEL

ut.set_seed(cf.SEED)
train_losses, val_losses = train(model, best_model_file, train_loader, val_loader, fa_ssim=cf.FA_SSIM, two_head=cf.TWO_HEAD)
