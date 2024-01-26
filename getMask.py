import torch
import ipdb

def getMask01(predMask, superPixelLabel, args):
    value1 = torch.tensor(1).to(args.device, dtype=torch.long)
    value0 = torch.tensor(0).to(args.device, dtype=torch.long)
    value_mask = torch.tensor(1).to(args.device, dtype=torch.float)
    
    N, H, W = superPixelLabel.shape
    # Initialize masks
    Mask01 = torch.zeros((N, H, W)).to(device=args.device)
    # Mask012 = torch.full((N, H, W), fill_value=2, device=args.device, dtype=torch.long)
    Mask012 = predMask.clone()
    
    
    # Iterate over each batch
    for batch in range(N):
        # Superpixel label contains Total_block
        SuperPixel_batch = superPixelLabel[batch]
        Total_block = torch.max(SuperPixel_batch)
        
        # Iterate over each superpixel block
        for block in range(1, Total_block + 1):
            # Get positions of pixels with the corresponding superpixel label
            indices = (SuperPixel_batch == block).nonzero()
            # Count total pixels
            Total_pixel = indices.size(0)
            # Count foreground pixels
            Total_fore = torch.sum(predMask[batch][indices[:, 0], indices[:, 1]])
            # Calculate percentage
            percentage = Total_fore / Total_pixel
            
            # Modify predMask and masks based on percentage
            if percentage >= args.threshold:
                Mask012[batch][indices[:, 0], indices[:, 1]] = value1
                Mask01[batch][indices[:, 0], indices[:, 1]] = value_mask
            elif percentage <= 1 - args.threshold:
                Mask012[batch][indices[:, 0], indices[:, 1]] = value0
                Mask01[batch][indices[:, 0], indices[:, 1]] = value_mask
    
    return Mask01, Mask012
