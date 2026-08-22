import torch

# File Options
T1W_FILE_NAME        = "T1_0_brain.nii.gz"
FA_FILE_NAME         = "T1_1_FA.nii.gz"
MD_FILE_NAME         = "T1_2_MD.nii.gz"
MASK_FILE_NAME       = "T1_mask_0_brain.nii.gz"
PRED_FA_NAME         = "Pred_FA.nii.gz"
PRED_MD_NAME         = "Pred_MD.nii.gz"

# Path Options
#TRAIN_DIR           = "/prject/data/train"
#VALIDATION_DIR      = "/project/data/val"
#TEST_DIR            = "/project/data/test"
BEST_MODEL_TRANSUNET = "best_model_transunet.pt"
BEST_MODEL_UNET      = "best_model_unet.pt"
BEST_MODEL_CNN       = "best_model_cnn.pt"

# Configuration
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE           = 10
EPOCHS               = 60
WARMUP_EPOCHS        = 10

# Options
SEED                 = 356          # 42, 356, 2026
BLOCK_SIZE           = 64           # 64, 80 
TWO_HEAD             = True         # True, False
FA_SSIM              = 0.0          # 0.0, 0.3
MODEL                = "TransUNet"  # TransUNet, UNet, CNN  
