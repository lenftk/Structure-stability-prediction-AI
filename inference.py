import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import StructureDataset
from model import StructureStabilityModel
from model_advanced import AdvancedStructureModel
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Inference Structure Stability Model")
    parser.add_argument("--data_dir", type=str, default="open", help="Path to open dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--backbone", type=str, default="resnet34", help="Model backbone")
    parser.add_argument("--advanced", action="store_true", help="Use Advanced Model architecture")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to best model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default="submission.csv", help="Submission CSV filename")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Paths
    test_csv = os.path.join(args.data_dir, "sample_submission.csv")
    test_img_dir = os.path.join(args.data_dir, "test")
    
    # Dataset and Loader
    # Ensure sample_submission is passed in without 'label' column causing issues
    test_dataset = StructureDataset(test_csv, test_img_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    if args.advanced:
        model = AdvancedStructureModel(backbone_name=args.backbone, pretrained=False)
    else:
        model = StructureStabilityModel(backbone_name=args.backbone, pretrained=False)
        
    model.load_state_dict(torch.load(args.weights_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            sample_ids = batch['id']
            front = batch['front'].to(args.device)
            top = batch['top'].to(args.device)
            
            logits = model(front, top)
            # Softmax to get probabilities
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
            for i in range(len(sample_ids)):
                # Class 0: unstable, Class 1: stable
                results.append({
                    'id': sample_ids[i],
                    'unstable_prob': probs[i][0],
                    'stable_prob': probs[i][1]
                })
                
    # Save submission
    # Submission format: id, unstable_prob, stable_prob
    df_sub = pd.DataFrame(results)
    df_sub = df_sub[['id', 'unstable_prob', 'stable_prob']] # Ensure column order
    df_sub.to_csv(args.output_file, index=False)
    print(f"Submission saved to {args.output_file}")

if __name__ == "__main__":
    main()
