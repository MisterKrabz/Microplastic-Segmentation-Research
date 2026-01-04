import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sam2.build_sam import build_sam2

MODEL_CONFIG = "sam2/sam2_hiera_l" 
CHECKPOINT = "sam2.1_hiera_large.pt"
DATA_DIR = "/staging/pwang384/URS_Project/ready_to_train_dataset"
BATCH_SIZE = 1
LR = 1e-5        
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {DEVICE}")

# --- DATASET ---
class MicroplasticDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        if not os.path.exists(self.img_dir): raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        self.images = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base_name + ".png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (1024, 1024))
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0:
            pad = np.random.randint(0, 20)
            bbox = [
                max(0, np.min(x_indices) - pad), 
                max(0, np.min(y_indices) - pad), 
                min(1024, np.max(x_indices) + pad), 
                min(1024, np.max(y_indices) + pad)
            ]
        else:
            bbox = [0, 0, 1024, 1024]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).float() / 255.0
        mask = mask.unsqueeze(0) 
        
        return image, mask, torch.tensor(bbox).float()

# --- MULTI-SCALE ADAPTER (The Permanent Fix) ---
class MultiScaleAdapter(nn.Module):
    """
    Adapts the Large Backbone features (256 dim) to the Tiny Decoder expectations.
    - Scale 0 (feat_s0): 256 -> 32
    - Scale 1 (feat_s1): 256 -> 64
    """
    def __init__(self, in_channels=256):
        super().__init__()
        self.proj_s0 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.proj_s1 = nn.Conv2d(in_channels, 64, kernel_size=1)
    
    def forward(self, features_list):
        # We expect a list of features. We project the first two.
        # features_list[0] corresponds to high-res (s0)
        # features_list[1] corresponds to next level (s1)
        s0 = self.proj_s0(features_list[0])
        s1 = self.proj_s1(features_list[1])
        return [s0, s1]

# --- LOSS ---
def criterion(pred_masks, gt_masks):
    # WEIGHTED LOSS: We tell the model that Plastic pixels are 20x more important than Background
    # Calculate weight based on class imbalance roughly (plastic is rare)
    pos_weight = torch.tensor([20.0]).to(DEVICE)
    # Weighted BCE
    bce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks, pos_weight=pos_weight)
    # Dice Loss (Standard overlap metric)
    pred_probs = torch.sigmoid(pred_masks)
    intersection = (pred_probs * gt_masks).sum()
    dice = 1 - (2. * intersection + 1.) / (pred_probs.sum() + gt_masks.sum() + 1.)
    return bce + dice

def main():
    print("Initializing Dataset...")
    train_ds = MicroplasticDataset(DATA_DIR, split="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print(f"Loading SAM2 Model...")
    model = build_sam2(MODEL_CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)
    model.train()

    # --- ADAPTER SETUP ---
    # We initialize the adapter to handle BOTH dimension mismatches (32 and 64)
    adapter = MultiScaleAdapter(in_channels=256).to(DEVICE)
    print("✅ Multi-Scale Adapter Initialized (256 -> 32/64)")

    # Freeze Backbone
    for name, param in model.named_parameters():
        if "mask_decoder" in name or "prompt_encoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False 

    # Optimizer updates model AND adapter
    params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters())) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=LR)
    
    scaler = torch.cuda.amp.GradScaler()

    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (images, masks, bboxes) in enumerate(train_loader):
            images, masks, bboxes = images.to(DEVICE), masks.to(DEVICE), bboxes.to(DEVICE)

            with torch.cuda.amp.autocast():
                features = model.image_encoder(images)
                sparse, dense = model.sam_prompt_encoder(
                    points=None,
                    boxes=bboxes.unsqueeze(1),
                    masks=None,
                )
                
                # Get raw features (256 dim)
                fpn_feats = features["backbone_fpn"]
                
                # [FIX] Apply MultiScale Adapter
                # Converts 256 -> 32 and 256 -> 64
                adapted_feats = adapter(fpn_feats)

                low_res_masks, _, _, _ = model.sam_mask_decoder(
                    image_embeddings=features["vision_features"],
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=adapted_feats # Pass ADAPTED features
                )

                upscaled_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
                loss = criterion(upscaled_masks, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            if i % 10 == 0: print(f"Epoch {epoch} Batch {i} Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{EPOCHS} Avg Loss: {epoch_loss / len(train_loader):.4f}")
        
        if (epoch+1) % 10 == 0:
             torch.save(model.state_dict(), f"sam2_epoch_{epoch+1}.pt")

    torch.save(model.state_dict(), "sam2_final.pt")
    torch.save(adapter.state_dict(), "adapter_final.pt")
    print("Training Complete.")

if __name__ == "__main__":
    main()
