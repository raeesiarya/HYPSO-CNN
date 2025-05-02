import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libraries import *
from functions.processing import load_label

#######################################################################################
#######################################################################################
#######################################################################################

def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
               save_path="models/best_model.pth", num_epochs=30):
    """
    Train the model using the provided training and validation data loaders.
    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to perform training on (CPU or GPU).
        save_path (str): Path to save the best model.
        num_epochs (int): Number of epochs to train the model.
    """
    best_accuracy = 0.0

    classes = ["Cloud", "Land", "Sea"]
    print(f"DEBUG - Number of training batches: {len(train_loader)}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        labels_per_epoch = []
        predictions_per_epoch = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, colour="red")

        # Training
        train_accuracy, total_loss = train_subloop(
            loop, model, criterion, optimizer, device,
            predictions_per_epoch, labels_per_epoch, total_loss
            )

        # Evaluation
        model.eval()
        all_labels_eval = []
        all_preds_eval = []

        with torch.no_grad():
            all_labels_eval, all_preds_eval, val_loss = eval_subloop(
                val_loader, model, criterion, device,
                all_labels_eval, all_preds_eval
                )

        val_accuracy = accuracy_score(all_labels_eval, all_preds_eval)

        # Print results
        print("Unique labels in eval-data:", sorted(set(all_labels_eval)))
        print("Unique predictions:", sorted(set(all_preds_eval)))
        print(colored(f"TRAIN - Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {train_accuracy:.2f}%", "magenta"))
        print(colored(f"EVAL  - Epoch {epoch+1}, Accuracy: {val_accuracy*100:.2f}%", "green"))
        print(classification_report(all_labels_eval, all_preds_eval, target_names=classes))

        # Log MLflow
        log_mlflow_train(train_accuracy, val_accuracy, total_loss, val_loss, epoch)
        log_classification_report_mlflow(classification_report(all_labels_eval, all_preds_eval,
                                                                target_names=classes, output_dict=True),
                                                                epoch)

        # Save model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, best_accuracy, save_path)
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")

        # Confusion Matrix
        confusion_matrix_plot(all_labels_eval, all_preds_eval, classes, epoch)

        scheduler.step()
        print(colored(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}", "yellow"))

        print("\n")

#######################################################################################

def train_subloop(loop, model, criterion, optimizer, device,
                  predictions_per_epoch, labels_per_epoch, total_loss):
    """
    Train the model for one epoch.
    Args:
        loop (tqdm): Progress bar for the training loop.
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to perform training on (CPU or GPU).
        predictions_per_epoch (list): List to store predictions for the epoch.
        labels_per_epoch (list): List to store labels for the epoch.
        total_loss (float): Total loss for the epoch.
    Returns:
        train_accuracy (float): Training accuracy for the epoch.
        total_loss (float): Total loss for the epoch.
    """
    correct = 0
    total = 0
    for batch in loop:
        spectrum, labels = batch
        spectrum = spectrum.unsqueeze(1).to(device, non_blocking=True)
        labels = labels.view(-1).to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(spectrum)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        predictions_per_epoch.append(output.detach().cpu())
        labels_per_epoch.append(labels.detach().cpu())

        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    
    return train_accuracy, total_loss

#######################################################################################

def eval_subloop(val_loader, model, criterion, device,
                 all_labels_eval, all_preds_eval):
    """
    Evaluate the model on the validation dataset.
    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The model to be evaluated.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to perform evaluation on (CPU or GPU).
        all_labels_eval (list): List to store labels for evaluation.
        all_preds_eval (list): List to store predictions for evaluation.
    Returns:
        all_labels_eval (list): List of labels for evaluation.
        all_preds_eval (list): List of predictions for evaluation.
        val_loss (float): Total loss for the validation dataset.
    """
    val_loss = 0.0
    for spectrum, labels in tqdm(val_loader, desc="Evaluation", leave=True, colour="blue"):
        spectrum = spectrum.unsqueeze(1).to(device)
        labels = labels.view(-1).to(device)
        output = model(spectrum)
        _, predicted = torch.max(output, 1)
        loss = criterion(output, labels)

        all_labels_eval.extend(labels.cpu().numpy())
        all_preds_eval.extend(predicted.cpu().numpy())
        val_loss += loss.item()
    
    return all_labels_eval, all_preds_eval, val_loss

#######################################################################################

def confusion_matrix_plot(all_labels_eval, all_preds_eval, classes, epoch):
    """
    Plot and save the confusion matrix.
    Args:
        all_labels_eval (list): List of labels for evaluation.
        all_preds_eval (list): List of predictions for evaluation.
        classes (list): List of class names.
        epoch (int): Current epoch number.
    """
    cm = confusion_matrix(all_labels_eval, all_preds_eval)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("Real Labels")
    plt.title("Confusion Matrix")

    output_dir = "plots/validation_plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"confusion_matrix_EPOCH_{epoch+1}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()

    mlflow.log_artifact(filepath, artifact_path="confusion_matrices")

#######################################################################################

def save_model(model, best_accuracy, save_path):
    """
    Save the model state dictionary and best accuracy to a file.
    Args:
        model (torch.nn.Module): The model to be saved.
        best_accuracy (float): Best accuracy achieved during training.
        save_path (str): Path to save the model.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_accuracy': best_accuracy
    }, save_path)

#######################################################################################

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Args:
        alpha (torch.Tensor, optional): Class weights for each class. If None, no weights are applied.
        gamma (float, optional): Focusing parameter. Default is 2.
        reduction (str, optional): Reduction method. Options are 'none', 'mean', or 'sum'.
            Default is 'mean'.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Initializes the Focal Loss.
        Args:
            alpha (torch.Tensor, optional): Class weights for each class. If None, no weights are applied.
            gamma (float, optional): Focusing parameter. Default is 2.
            reduction (str, optional): Reduction method. Options are 'none', 'mean', or 'sum'.
                Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the Focal Loss.
        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth labels.
        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha = self.alpha.gather(0, targets)
            focal_loss = alpha * (1 - p_t) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
#######################################################################################

def get_class_weights(label_paths, num_classes):
    """
    Calculate class weights based on the frequency of each class in the dataset.
    Args:
        label_paths (list): List of paths to label files.
        num_classes (int): Number of classes in the dataset.
    Returns:
        torch.Tensor: Class weights for each class.
    """
    all_labels = []

    for path in label_paths:
        labels = load_label(path)
        all_labels.extend(labels)

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()

    # No zero division
    epsilon = 1e-6
    class_counts += epsilon

    print("Using class weights.")

    # Calculate weights
    alpha = (1.0 / class_counts)
    alpha = alpha / alpha.sum() # Normalization

    print(colored(f"Successfully calculated class weights: {alpha}", "green"))

    return alpha

#######################################################################################

def log_mlflow_pre_train(EPOCHS, BATCH_SIZE, LR, LABEL_SMOOTHING,
               KERNEL_SIZE, STARTING_KERNELS, NUM_FEATURES, NUM_CLASSES,
               optimizer, scheduler, criterion, model, train_dataset, eval_dataset):
    """
    Log hyperparameters and model information to MLflow.
    Args:
        EPOCHS (int): Number of epochs for training.
        BATCH_SIZE (int): Batch size for training.
        LR (float): Learning rate for the optimizer.
        LABEL_SMOOTHING (float): Label smoothing parameter.
        KERNEL_SIZE (int): Kernel size for the model.
        STARTING_KERNELS (int): Number of starting kernels for the model.
        NUM_FEATURES (int): Number of features in the input data.
        NUM_CLASSES (int): Number of classes in the output data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler used for training.
        criterion (torch.nn.Module): Loss function used for training.
        model (torch.nn.Module): Model architecture used for training.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        eval_dataset (torch.utils.data.Dataset): Evaluation dataset.
    """
    mlflow.log_param("EPOCHS", EPOCHS)
    mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
    mlflow.log_param("LR", LR)
    mlflow.log_param("LABEL_SMOOTHING", LABEL_SMOOTHING)
    mlflow.log_param("KERNEL_SIZE", KERNEL_SIZE)
    mlflow.log_param("STARTING_KERNELS", STARTING_KERNELS)
    mlflow.log_param("NUM_FEATURES", NUM_FEATURES)
    mlflow.log_param("NUM_CLASSES", NUM_CLASSES)
    mlflow.log_param("optimizer", optimizer.__class__.__name__)
    mlflow.log_param("scheduler", scheduler.__class__.__name__)
    mlflow.log_param("loss_function", criterion.__class__.__name__)
    mlflow.log_param("model", model.__class__.__name__)
    mlflow.log_param("train_dataset_size", len(train_dataset)/653016)
    mlflow.log_param("eval_dataset_size", len(eval_dataset)/653016)

#######################################################################################

def log_mlflow_train(train_accuracy, val_accuracy, train_loss, val_loss, epoch):
    """
    Log training and validation metrics to MLflow.
    Args:
        train_accuracy (float): Training accuracy for the current epoch.
        val_accuracy (float): Validation accuracy for the current epoch.
        train_loss (float): Training loss for the current epoch.
        val_loss (float): Validation loss for the current epoch.
        epoch (int): Current epoch number.
    """
    mlflow.log_metric(f"train_accuracy", train_accuracy/100, step=epoch)
    mlflow.log_metric(f"val_accuracy", val_accuracy, step=epoch)
    mlflow.log_metric(f"train_loss", train_loss, step=epoch)
    mlflow.log_metric(f"val_loss", val_loss, step=epoch)

#######################################################################################

def log_classification_report_mlflow(report_dict, epoch):
    """
    Log classification report metrics to MLflow.
    Args:
        report_dict (dict): Classification report dictionary containing metrics.
        epoch (int): Current epoch number.
    """
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if "support" or "accuracy" not in metric_name:
                    mlflow.log_metric(f"{class_name}_{metric_name}", value, step=epoch)
        else:
            if "support" or "accuracy" not in class_name:
                mlflow.log_metric(f"{class_name}", metrics, step=epoch)