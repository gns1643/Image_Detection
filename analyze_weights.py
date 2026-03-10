import torch
import os

weights_path = r"C:\Users\JIHOON\Documents\GitHub\Image_Detection\modules\models\apdrawing_style\netG_A_latest.pth"

if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at {weights_path}")
else:
    print(f"Analyzing weights: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    
    keys = list(state_dict.keys())
    print(f"Total keys: {len(keys)}")
    print("\nFirst 20 keys:")
    for k in keys[:20]:
        print(f"  {k}")
        
    print("\nLast 5 keys:")
    for k in keys[-5:]:
        print(f"  {k}")
