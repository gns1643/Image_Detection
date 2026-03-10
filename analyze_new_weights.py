import torch
import os

weights_path = r"C:\Users\JIHOON\Documents\GitHub\Image_Detection\modules\models\checkpoints\checkpoints\apdrawinggan++_author\150_net_gen.pt"

if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at {weights_path}")
else:
    print(f"Analyzing weights: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    
    # If it's a checkpoint, it might have 'state_dict' key
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        print("Found 'state_dict' key in checkpoint.")
    
    keys = list(state_dict.keys())
    print(f"Total keys: {len(keys)}")
    print("\nFirst 20 keys:")
    for k in keys[:20]:
        print(f"  {k}")
