import os
import json
import torch
import matplotlib.pyplot as plt
from dataset import SaliencyDataset
from model_resnet18 import BiologicallyOptimizedSaliencyNet
from model_resnet50 import ResNet50SaliencyNet

def plot_reconstruction(sample_idx=42, dataset_type="val", weights_filename="best_model_resnet18_baseline.pth", save_filename=None):
    """
    Loads a trained model checkpoint and plots a side-by-side comparison 
    of the Input, Human Ground Truth, and AI Prediction. Optionally saves to disk.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "cv2_project_data")
    weights_path = os.path.join(base_dir, weights_filename)
    
    images_txt = f"{dataset_type}_images.txt"
    fixations_txt = f"{dataset_type}_fixations.txt"
    
    dataset = SaliencyDataset(data_dir, images_txt, fixations_txt)
    if sample_idx >= len(dataset):
        print(f"Error: Index {sample_idx} out of range for {dataset_type} dataset.")
        return
        
    image_tensor, true_fixation = dataset[sample_idx]
    
    # Dynamically select architecture based on file naming convention
    if "resnet50" in weights_filename.lower():
        model = ResNet50SaliencyNet().to(device)
    else:
        model = BiologicallyOptimizedSaliencyNet().to(device)
        
    if not os.path.exists(weights_path):
        print(f"Error: Weight checkpoint file not found at {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        input_batch = image_tensor.unsqueeze(0).to(device)
        predicted_tensor = model(input_batch).squeeze(0).cpu()
        
    img_to_show = image_tensor.permute(1, 2, 0).numpy()
    img_to_show = img_to_show * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img_to_show = img_to_show.clip(0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_to_show)
    axes[0].set_title(f"Input Image ({dataset_type.capitalize()} #{sample_idx})", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(true_fixation.squeeze(0).numpy(), cmap='gray')
    axes[1].set_title("Ground Truth (Human Gaze)", fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(predicted_tensor.squeeze(0).numpy(), cmap='gray')
    axes[2].set_title("AI Predicted Saliency", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_filename:
        export_path = os.path.join(base_dir, save_filename)
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        print(f" Successfully saved reconstruction figure to: {export_path}")
        
    plt.show()


def plot_saved_loss(experiment_name, save_filename=None):
    """
    Loads historical metric JSON logs for a single experiment and plots 
    the training vs validation curves immediately without needing to retrain.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    metrics_file = os.path.join(base_dir, f"metrics_{experiment_name}.json")
    
    if not os.path.exists(metrics_file):
        print(f"Error: Metric file '{metrics_file}' not found.")
        return

    with open(metrics_file, "r") as f:
        log_data = json.load(f)
    
    train_loss = log_data.get("train_history") or log_data.get("train_loss")
    val_loss = log_data.get("val_history") or log_data.get("val_loss")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss', color='dodgerblue', linewidth=2, marker='o', markersize=4)
    plt.plot(val_loss, label='Validation Loss', color='crimson', linewidth=2, marker='x', markersize=4)
    
    plt.title(f"Loss Convergence: {experiment_name.replace('_', ' ').title()}", fontsize=13, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Loss Value', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if save_filename:
        export_path = os.path.join(base_dir, save_filename)
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        print(f" Saved single loss curve chart to: {export_path}")
        
    plt.show()


def plot_combined_validation(experiment_list, save_filename="comprehensive_experiment_comparison.png"):
    """
    Scans the repository for multiple JSON experiment entries and overlays 
    their validation tracks onto a single comparison chart for report documentation.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    formatting = {
        "resnet18_baseline": ("teal", ":", "o"),
        "resnet18_regularized": ("darkviolet", "-", "s"),
        "resnet50_hybrid_run": ("limegreen", "-", "^")
    }

    plt.figure(figsize=(12, 6.5))
    plots_added = False

    for exp in experiment_list:
        metrics_file = os.path.join(base_dir, f"metrics_{exp}.json")
        
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                data = json.load(f)
                
            val_loss = data.get("val_history") or data.get("val_loss")
            if val_loss:
                # Fallback style configuration if using a custom unique experiment name string
                color, style, marker = formatting.get(exp, ("black", "-", "o"))
                plt.plot(val_loss, label=f"{exp} (Val)", color=color, linestyle=style, marker=marker, markersize=5, linewidth=2)
                plots_added = True

    if plots_added:
        plt.title("Comparative Dashboard: Model Validation Tracking", fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11, loc='upper right')
        plt.tight_layout()
        
        export_path = os.path.join(base_dir, save_filename)
        plt.savefig(export_path, dpi=300)
        print(f" Comparative report chart saved to: {export_path}")
        plt.show()
    else:
        print("Execution stopped: No valid experiment metric logs found.")