"""
Visualization utilities for the food weight prediction pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import cv2


def visualize_image_pair(before_path: str, after_path: str,
                        before_weight: float, after_weight: float,
                        predicted_weight: float = None,
                        save_path: str = None):
    """
    Visualize a before-after image pair with weights.

    Args:
        before_path: Path to before image
        after_path: Path to after image
        before_weight: Weight before eating (grams)
        after_weight: Actual weight after eating (grams)
        predicted_weight: Predicted weight after eating (grams)
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Load images
    img_before = cv2.imread(before_path)
    img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)

    img_after = cv2.imread(after_path)
    img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)

    # Display before image
    axes[0].imshow(img_before)
    axes[0].set_title(f'Before Eating\nWeight: {before_weight:.1f}g', fontsize=12)
    axes[0].axis('off')

    # Display after image
    axes[1].imshow(img_after)

    title = f'After Eating\nActual: {after_weight:.1f}g'
    if predicted_weight is not None:
        error = abs(predicted_weight - after_weight)
        title += f'\nPredicted: {predicted_weight:.1f}g\nError: {error:.1f}g'

    axes[1].set_title(title, fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()


def visualize_segmentation(image: np.ndarray, mask: np.ndarray,
                          masked_image: np.ndarray = None,
                          save_path: str = None):
    """
    Visualize segmentation results.

    Args:
        image: Original image
        mask: Segmentation mask
        masked_image: Image with mask applied (optional)
        save_path: Path to save figure (optional)
    """
    num_plots = 3 if masked_image is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 5, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Mask
    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    # Masked image
    if masked_image is not None:
        axes[2].imshow(masked_image)
        axes[2].set_title('Segmented Food')
        axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_augmentations(original_image: np.ndarray,
                           augmented_images: List[np.ndarray],
                           titles: List[str] = None,
                           save_path: str = None):
    """
    Visualize data augmentation examples.

    Args:
        original_image: Original image
        augmented_images: List of augmented images
        titles: List of titles for each augmentation
        save_path: Path to save figure (optional)
    """
    num_images = len(augmented_images) + 1
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if num_images > 1 else [axes]

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Augmented images
    for i, aug_img in enumerate(augmented_images):
        axes[i + 1].imshow(aug_img)
        title = titles[i] if titles and i < len(titles) else f'Augmentation {i+1}'
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')

    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_training_history(history: Dict, metrics: List[str] = None,
                          save_path: str = None):
    """
    Plot training history.

    Args:
        history: Training history dictionary
        metrics: List of metrics to plot (default: all available)
        save_path: Path to save figure (optional)
    """
    if metrics is None:
        # Auto-detect metrics (exclude validation metrics)
        metrics = [k for k in history.keys() if not k.startswith('val_')]

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        train_values = history.get(metric, [])
        val_values = history.get(f'val_{metric}', [])

        epochs = range(1, len(train_values) + 1)

        axes[i].plot(epochs, train_values, 'b-', label=f'Training {metric}')
        if val_values:
            axes[i].plot(epochs, val_values, 'r-', label=f'Validation {metric}')

        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'{metric.capitalize()} over Epochs')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray,
                           save_path: str = None):
    """
    Plot regression results with scatter plot and error distribution.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('True Weight (g)')
    axes[0].set_ylabel('Predicted Weight (g)')
    axes[0].set_title('Predicted vs True Weight')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = y_pred - y_true
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', lw=2, label='Zero error')
    axes[1].set_xlabel('Prediction Error (g)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    r2 = 1 - (np.sum(errors**2) / np.sum((y_true - y_true.mean())**2))

    stats_text = f'MAE: {mae:.2f}g\nRMSE: {rmse:.2f}g\nR²: {r2:.4f}'
    axes[1].text(0.05, 0.95, stats_text,
                transform=axes[1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: List[str] = None,
                         save_path: str = None):
    """
    Plot confusion matrix for classification results.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure (optional)
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names if class_names else 'auto',
               yticklabels=class_names if class_names else 'auto')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def create_inference_report(image_before: str, image_after: str,
                           food_category: str, predicted_weight: float,
                           actual_weight: float = None,
                           save_path: str = None):
    """
    Create a comprehensive inference report with visualizations.

    Args:
        image_before: Path to before image
        image_after: Path to after image
        food_category: Predicted food category
        predicted_weight: Predicted weight
        actual_weight: Actual weight (if available)
        save_path: Path to save report
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    img_before = cv2.imread(image_before)
    img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
    img_after = cv2.imread(image_after)
    img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)

    ax1.imshow(img_before)
    ax1.set_title('Before Eating', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(img_after)
    ax2.set_title('After Eating', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Prediction results
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    results_text = f"""
    PREDICTION RESULTS
    {'=' * 60}

    Food Category: {food_category}
    Predicted Weight: {predicted_weight:.2f}g
    """

    if actual_weight is not None:
        error = abs(predicted_weight - actual_weight)
        error_pct = (error / actual_weight) * 100
        results_text += f"""
    Actual Weight: {actual_weight:.2f}g
    Absolute Error: {error:.2f}g
    Percentage Error: {error_pct:.2f}%
        """

    ax3.text(0.1, 0.5, results_text, fontsize=12, family='monospace',
            verticalalignment='center')

    plt.suptitle('Food Weight Prediction Report', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
