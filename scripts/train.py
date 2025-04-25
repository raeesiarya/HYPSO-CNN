import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libraries import *
from manage_data import read_csv_file
from dataset import merged_hyperspectral_dataset
from functions.processing import normalization_manager
from functions.train_functions import train_loop
from functions.train_functions import get_class_weights
from functions.train_functions import FocalLoss
from functions.train_functions import log_mlflow_pre_train
from models.cnn_1d import JustoLiuNet1D_torch

#######################################################################################
#######################################################################################
#######################################################################################

# MLflow
mlflow.set_experiment("CNN_hyperspectral_v1")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001
LABEL_SMOOTHING = 0.1
KERNEL_SIZE = 6
STARTING_KERNELS = 6
NUM_FEATURES = 114
NUM_CLASSES = 3

with mlflow.start_run():

    # Data
    train_bip_paths, train_labels_paths, _ = read_csv_file("csv/train_files.csv")
    eval_bip_paths, eval_labels_paths, _ = read_csv_file("csv/evaluate_files.csv")

    # Normalizer
    normalizer = normalization_manager()
    raw_train = merged_hyperspectral_dataset(train_bip_paths, train_labels_paths)
    normalizer.fit(raw_train.images)
    train_dataset = merged_hyperspectral_dataset(train_bip_paths, train_labels_paths, normalizer)
    eval_dataset = merged_hyperspectral_dataset(eval_bip_paths, eval_labels_paths, normalizer)

    # Dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=8,
                             pin_memory=True)

    # Model
    model = JustoLiuNet1D_torch(num_features=NUM_FEATURES, num_classes=NUM_CLASSES,
                                 kernel_size=KERNEL_SIZE, starting_kernels=STARTING_KERNELS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(colored(f"Total parameters in {model.__class__.__name__}: {total_params}", "magenta"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    alpha = get_class_weights(train_labels_paths, NUM_CLASSES)
    #criterion = nn.CrossEntropyLoss(weight=alpha.to(device), label_smoothing=LABEL_SMOOTHING).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(device)
    #criterion = FocalLoss(alpha=alpha.to(device), gamma=3, reduction='sum').to(device)

    # Logging
    log_mlflow_pre_train(EPOCHS, BATCH_SIZE, LR, LABEL_SMOOTHING,
               KERNEL_SIZE, STARTING_KERNELS, NUM_FEATURES, 
               NUM_CLASSES,optimizer, scheduler, criterion, 
               model, train_dataset, eval_dataset)

    # Training
    print("Starting training...")
    train_loop(model, train_loader, eval_loader, criterion, 
               optimizer, scheduler, device, num_epochs=EPOCHS)
    print("Training finished.")