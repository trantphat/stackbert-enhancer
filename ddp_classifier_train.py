import os
import time
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer
from transformers import logging as transformer_logging

from model import BertCustomBinaryClassifier
from utils.data_preprocessing import load_dataset
from utils.evaluate_metrics import evaluate_metrics

# Suppress warnings from transformers library
transformer_logging.set_verbosity_error()


def setup_logging(log_dir="logs", rank=0):
    """Sets up logging to output only on rank 0 and saves logs to a specified directory."""
    # Create directory name based on the current date
    date_dir = datetime.now().strftime("%Y-%m-%d")
    full_log_dir = os.path.join(log_dir, date_dir)

    # Ensure the log directory exists
    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir)

    # Generate log filename with date and time
    log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(full_log_dir, log_filename)
    logger = logging.getLogger("training")

    # Only setup file handler for rank 0
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()]
        )
    else:
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()]
        )

    return logger


def setup_DDP_mp(init_method, local_rank, rank, world_size, logger, backend="nccl", verbose=False):
    """Initializes the distributed process group and sets the device."""
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        file_name = os.path.basename(__file__)
        logger.info(f"[Initialization] File: {file_name}")
        logger.info(f"[Initialization] Using device: {device}")
        logger.info(f"[Initialization] Local Rank: {local_rank} | Global Rank: {rank} | World Size: {world_size}")
    return device


def gather_results(tensor):
    """Gathers tensors from all ranks and returns a concatenated tensor."""
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors)


def gather_loss(loss, device):
    """Gathers and sums losses from all GPUs."""
    loss_tensor = torch.tensor([loss], dtype=torch.float32, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor.item()


def train(dataloader, model, optimizer, device, lambd, train_loss, real_labels, pre_labels, k):
    """Trains the model on the given dataloader."""
    for batch_data in dataloader:
        model.train()
        optimizer.zero_grad()

        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        labels = batch_data["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, kmer=k)
        loss = outputs[0]
        lambd_tensor = torch.tensor(lambd, requires_grad=True).to(device)  # Ensure this is on the correct device

        # L2-normalization
        L2_loss = torch.tensor(0.0, requires_grad=True).to(device)
        for param in model.parameters():
            L2_loss += torch.norm(param, p=2)
        loss += lambd_tensor * L2_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        train_loss += loss.item()

        logits = outputs[1].detach()

        if len(real_labels) == 0:
            real_labels = labels
            pre_labels = logits
        else:
            real_labels = torch.cat([real_labels, labels], dim=0)
            pre_labels = torch.cat([pre_labels, logits], dim=0)

    # Return the accumulated loss and the concatenated labels and logits
    return train_loss, real_labels, pre_labels


def evaluate(dataloader, model, device, test_loss, real_labels, pre_labels):
    """Evaluates the model on the given dataloader."""

    for batch_data in dataloader:
        model.eval()
        with torch.no_grad():
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            test_loss += loss.item()

            logits = outputs[1]

            if len(real_labels) == 0:
                real_labels = labels
                pre_labels = logits
            else:
                real_labels = torch.cat([real_labels, labels], dim=0)
                pre_labels = torch.cat([pre_labels, logits], dim=0)

    return test_loss, real_labels, pre_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes for distributed training")
    parser.add_argument("--ngpus_per_node", default=4, type=int, help="Number of GPUs per node for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12355", type=str, help="URL used to set up distributed training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--node_rank", default=0, type=int, help="Node rank for distributed training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum sequence length for input data")
    parser.add_argument("--model_path", type=str, help="Path to the model directory")
    parser.add_argument("--test_data_path", type=str, help="Path to the test dataset file")
    parser.add_argument("--train_data_path", type=str, help="Path to the training dataset file")
    parser.add_argument("--save_model", action="store_true", help="Save the model checkpoint if set")
    return parser.parse_args()


def main(local_rank, ngpus_per_node, args):
    """Main training function."""
    args.local_rank = local_rank
    args.rank = args.node_rank * ngpus_per_node + local_rank

    logger = setup_logging(rank=args.rank)
    device = setup_DDP_mp(init_method=args.dist_url, local_rank=args.local_rank, rank=args.rank, world_size=args.world_size, verbose=True, logger=logger)

    # Parameters setup
    k_params = {
        3: {"learning_rate": 5e-05, "lambda": 9e-4},
        4: {"learning_rate": 5e-05, "lambda": 4e-4},  # 4: {"learning_rate": 5e-05, "lambda": 2e-4},
        5: {"learning_rate": 5e-05, "lambda": 1e-4},
        6: {"learning_rate": 5e-05, "lambda": 5e-4},
    }
    seed = 1337  # Random seed
    results = []  # Results tracking
    identifier_model_date = "2025-02-07"

    for k_target, params in k_params.items():
        learning_rate = params["learning_rate"]
        lambd = params["lambda"]

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # Log the current configuration of hyperparameters
        logger.info("Starting training for %s-mer classifier with: batch size=%s | learning rate=%s | lambda=%s", k_target, args.batch_size, learning_rate, lambd)

        # Define paths for model and data based on the current k-mer target
        args.model_path = os.path.join(".", "outputs", "identifier_models", f"{identifier_model_date}", f"{k_target}-mer")
        args.test_data_path = os.path.join(".", "data", "enhancer_classification", f"{k_target}-mer_classification_test.txt")
        args.train_data_path = os.path.join(".", "data", "enhancer_classification", f"{k_target}-mer_classification_train.txt")

        # Load training and testing datasets
        train_dataset = load_dataset(args, validation=False)
        test_dataset = load_dataset(args, validation=True)

        # Create distributed samplers for shuffling and splitting data across workers
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=args.world_size, rank=args.rank)
        test_sampler = DistributedSampler(test_dataset, shuffle=False, num_replicas=args.world_size, rank=args.rank)

        # Initialize data loaders for batch processing
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)

        # Load the pre-trained model and prepare it for distributed training
        model = BertCustomBinaryClassifier.from_pretrained(args.model_path, num_labels=1).to(device)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        tokenizer = BertTokenizer.from_pretrained(args.model_path)

        # Set up optimizer and learning rate scheduler
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        for epoch in range(args.epochs):
            # Ensure proper shuffling by setting the epoch for distributed samplers
            train_dataloader.sampler.set_epoch(epoch)
            test_dataloader.sampler.set_epoch(epoch)  # Optional for test sampler
            total_batch_size = len(train_dataloader) * args.world_size  # Total batch size across workers
            epoch_start_time = time.time()  # Record start time for the epoch

            # Train the model and compute training loss and predictions
            train_loss, true_labels_train, predicted_labels_train = train(
                dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                device=device,
                lambd=lambd,
                train_loss=0,
                real_labels=torch.tensor([], dtype=torch.long).to(device),
                pre_labels=torch.tensor([], dtype=torch.float).to(device),
                k=k_target,
            )

            # Gather results from all workers and calculate training accuracy
            true_labels_train, predicted_labels_train = gather_results(true_labels_train), gather_results(predicted_labels_train)
            train_accuracy, *_ = evaluate_metrics(predicted_labels_train.cpu().numpy(), true_labels_train.cpu().numpy())

            # Evaluate the model on the test dataset
            test_loss, true_labels_test, predicted_labels_test = evaluate(
                dataloader=test_dataloader,
                model=model,
                device=device,
                test_loss=0,
                real_labels=torch.tensor([], dtype=torch.long).to(device),
                pre_labels=torch.tensor([], dtype=torch.float).to(device),
            )

            # Gather results from all workers, convert tensors to numpy arrays, and calculate evaluation metrics
            true_labels_test, predicted_labels_test = gather_results(true_labels_test), gather_results(predicted_labels_test)
            test_accuracy, test_sn, test_sp, test_mcc, test_auc = evaluate_metrics(predicted_labels_test.cpu().numpy(), true_labels_test.cpu().numpy())

            # Update the learning rate scheduler after each epoch
            scheduler.step()

            # Log metrics every 10 epochs if this is the main process (rank 0)
            if dist.get_rank() == 0 and (epoch + 1) % 10 == 0:
                epoch_end_time = time.time()  # Record end time for the epoch
                logger.info(
                    f"Train Accuracy={train_accuracy:.4f}, "
                    f"Test Accuracy={test_accuracy:.4f}, "
                    f"Sn={test_sn:.4f}, "
                    f"Sp={test_sp:.4f}, "
                    f"MCC={test_mcc:.4f}, "
                    f"AUC={test_auc:.4f}, "
                    f"Time={epoch_end_time - epoch_start_time:.2f}s"
                )

        # Store results for this hyperparameter configuration
        results.append(
            {
                "k-mer": f"{k_target}-mer",
                "learning_rate": f"{learning_rate}",
                "lambda": f"{lambd}",
                "train_accuracy": f"{train_accuracy:.4f}",
                "test_accuracy": f"{test_accuracy:.4f}",
                "sn": f"{test_sn:.4f}",
                "sp": f"{test_sp:.4f}",
                "mcc": f"{test_mcc:.4f}",
                "auc": f"{test_auc:.4f}",
            }
        )

        # Save the model if enabled and on the main process only
        if args.save_model and dist.get_rank() == 0:  # Save on the main process only
            # Generate directory name based on the current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            model_save_dir = os.path.join("outputs", "classifier_models", current_date)
            os.makedirs(model_save_dir, exist_ok=True)

            # Define the model save path
            model_filename = f"{k_target}-mer"
            model_save_path = os.path.join(model_save_dir, model_filename)

            # Check if the model file already exists
            if os.path.exists(model_save_path):
                # Append timestamp to filename to avoid overwriting
                timestamp = datetime.now().strftime("%H%M%S")
                model_filename = f"{k_target}-mer_{timestamp}"
                model_save_path = os.path.join(model_save_dir, model_filename)

            # Save the model and tokenizer
            model.module.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logger.info(f"{k_target}-mer classifier model saved to {model_save_path}")

    # Save the results to a CSV file
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_save_dir = os.path.join("outputs", "results", current_date)
    os.makedirs(results_save_dir, exist_ok=True)

    # Define the CSV output path
    csv_filename = f"classifier_3456-mer_{current_timestamp}.csv"
    csv_output_path = os.path.join(results_save_dir, csv_filename)

    # Save the results DataFrame to a CSV file
    pd.DataFrame(results).to_csv(csv_output_path, index=False)
    logger.info(f"Results saved to CSV file at {csv_output_path}")
    logger.info("========== Done ==========")


if __name__ == "__main__":
    # Initialize some arguments
    args = parse_args()
    args.world_size = args.ngpus_per_node * args.nodes

    # Run with torch.multiprocessing
    mp.spawn(main, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
