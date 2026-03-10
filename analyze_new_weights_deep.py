import torch
import os

weights_path = r"C:\Users\JIHOON\Documents\GitHub\Image_Detection\modules\models\checkpoints\checkpoints\apdrawinggan++_author\150_net_gen.pt"

if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at {weights_path}")
else:
    print(f"Analyzing weights: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    
    if 'G' in state_dict:
        g_weights = state_dict['G']
        print(f"Total keys in 'G': {len(g_weights)}")
        keys = list(g_weights.keys())
        print("\nFirst 10 keys in 'G':")
        for k in keys[:10]:
            print(f"  {k}")
    else:
        print("Key 'G' not found!")
