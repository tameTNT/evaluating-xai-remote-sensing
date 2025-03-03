import argparse
import os
import platform
import time
import typing as t
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch
import torch.nn as nn
import wandb
from tqdm.autonotebook import tqdm

# for when on NCC to be able to import local packages
os.chdir(os.path.expanduser("~/l3_project"))

import dataset_processing
import helpers

# Create argument parser
parser = argparse.ArgumentParser(description="Train a model on a dataset.")
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="Specify a random PyTorch seed for reproducibility.",
)
parser.add_argument(
    "--checkpoints_root_name",
    type=Path,
    default="checkpoints",
    help="Specify the directory for checkpoints with l3_project.",
)
parser.add_argument(
    "--model_name",
    type=str,
    choices=helpers.models.MODEL_NAMES,
    help="Name of the model to train.",
    required=True,
)
parser.add_argument(
    "--dataset_name",
    type=str,
    choices=dataset_processing.core.DATASET_NAMES,
    help="Name of the dataset to train on.",
    required=True,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size to use for DataLoaders.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="Number of workers to use for DataLoaders.",
)
parser.add_argument(
    "--optimiser_name",
    type=str,
    default="SGD",
    choices=["SGD"],  # todo: support more optimisers
)
parser.add_argument(
    "--loss_criterion_name",
    type=str,
    default="CrossEntropyLoss",
    choices=["CrossEntropyLoss"],  # todo: support more loss criteria
    help="Loss criterion to use.",
)
parser.add_argument(
    "--frozen_lr",
    type=float,
    default=0.01,
    help="Learning rate to use for training the partially frozen model.",
)
parser.add_argument(
    "--frozen_max_epochs",
    type=int,
    default=20,
    help="Maximum number of epochs to train the partially frozen model.",
)
parser.add_argument(
    "--full_lr",
    type=float,
    default=0.001,
    help="Learning rate to use for training the full model.",
)
parser.add_argument(
    "--frozen_lr_early_stop_threshold",
    type=float,
    default=0.0001,
    help="Learning rate threshold for early stopping when training frozen model.",
)
parser.add_argument(
    "--full_max_epochs",
    type=int,
    default=50,
    help="Maximum number of epochs to train the full model.",
)
parser.add_argument(
    "--lr_early_stop_threshold",
    type=float,
    default=0.0001,
    help="Learning rate threshold for early stopping when training full model.",
)
parser.add_argument(
    "--wandb_track_run",
    action="store_true",
    help="Whether to track the run with wandb.",
)

# Parse arguments
args = parser.parse_args()
random_seed = args.random_seed
checkpoints_root_name = args.checkpoints_root_name
model_name = args.model_name
dataset_name = args.dataset_name

batch_size = args.batch_size
num_workers = args.num_workers
optimiser_name = args.optimiser_name
wandb_track_run = args.wandb_track_run
loss_criterion: nn.Module = getattr(nn, args.loss_criterion_name)()

frozen_lr = args.frozen_lr
frozen_lr_early_stop_threshold = args.frozen_lr_early_stop_threshold
frozen_max_epochs = args.frozen_max_epochs

full_lr = args.full_lr
lr_early_stop_threshold = args.lr_early_stop_threshold
full_max_epochs = args.full_max_epochs

# Actual script starts here
lg = helpers.logging.get_logger("main")
lg.debug("Successfully imported packages.")

if torch.cuda.is_available():
    torch_device = torch.device('cuda')
    lg.debug(f'Found {torch.cuda.get_device_name()} to use as a cuda device.')
elif platform.system() == 'Darwin':
    torch_device = torch.device('mps')
else:
    torch_device = torch.device('cpu')
lg.info(f'Using {torch_device} as torch device.')

if platform.system() != 'Linux':
    torch.set_num_threads(1)
    lg.debug('Set number of threads to 1 as using a non-Linux machine.')

np_rng = np.random.default_rng(random_seed)
_ = torch.manual_seed(random_seed)
lg.debug(f'Random seed set to {random_seed}.')

checkpoints_path = Path.home() / "l3_project" / checkpoints_root_name
checkpoints_path.mkdir(exist_ok=True)
lg.debug(f'Checkpoints directory set to {checkpoints_path.resolve()}.')


def get_dataset_object(
        name: dataset_processing.core.DATASET_NAMES,
        split: t.Literal["train", "val", "test"],
        image_size: int,
        download: bool = False,
        do_transforms: bool = True,
):
    kwargs = {
        "split": split,
        "image_size": image_size,
        "download": download,
        "do_transforms": do_transforms,
    }

    if name == "EuroSATRGB":
        lg.debug("Loading EuroSATRGB dataset...")
        ds = dataset_processing.eurosat.EuroSATRGB(**kwargs)
    elif name == "EuroSATMS":
        lg.debug("Loading EuroSATMS dataset...")
        ds = dataset_processing.eurosat.EuroSATMS(**kwargs)
    else:
        lg.error(f"Invalid dataset name ({name}) provided to get_dataset_object.")
        raise ValueError(f"Dataset {name} does not exist.")

    lg.info(f"Dataset {name} ({split}) loaded with {len(ds)} samples.")
    return ds


def get_model_type(
        name: helpers.models.MODEL_NAMES,
) -> t.Type[helpers.models.FreezableModel]:
    if name == "ResNet50":
        lg.debug("Returning ResNet50 model type...")
        m = helpers.models.FineTunedResNet50
    else:
        lg.error(f"Invalid model name ({name}) provided to get_model_type.")
        raise ValueError(f"Model {name} does not exist.")

    return m


model_type = get_model_type(model_name)

training_dataset = get_dataset_object(dataset_name, "train", model_type.expected_input_dim)
validation_dataset = get_dataset_object(dataset_name, "val", model_type.expected_input_dim)

model = model_type(
    n_input_bands=training_dataset.N_BANDS,
    n_output_classes=training_dataset.N_CLASSES
).to(torch_device)

training_dataloader = torch.utils.data.DataLoader(
    training_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True
)
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
)

validation_iterator = iter(dataset_processing.core.cycle(validation_dataloader))


def get_opt_and_scheduler(lr: float, reduction_steps: int = 4):
    opt: torch.optim.Optimizer = getattr(torch.optim, optimiser_name)(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-6, momentum=0.9, nesterov=True,
    )
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=np.float_power(10, -1 / reduction_steps),
        # requires reduction_steps reductions to reduce by factor 10 (*0.1)
        patience=5, threshold=0.005
    )
    return opt, sch


def train_model(
        lr: float,
        is_frozen_model: bool = False,
        max_epochs: int = 50,
        scheduler_reduction_steps: int = 4,
        early_stop_threshold: float = 0.0001
):
    weights_save_path = checkpoints_path / training_dataset.__class__.__name__ / model.__class__.__name__
    if is_frozen_model:
        weights_save_path /= "frozen_partial"
    else:
        weights_save_path /= "full"
    weights_save_path.mkdir(parents=True, exist_ok=True)
    lg.debug(f"Output path set to {weights_save_path.resolve()}.")

    optimiser, scheduler = get_opt_and_scheduler(lr, scheduler_reduction_steps)
    lg.debug(f"Initialised optimiser (lr={lr}) and scheduler.")

    if wandb_track_run:
        wandb_run = wandb.init(
            save_code=True,
            project="evaluating_xAI_for_RS",
            name=f"{dataset_name}_{model_name}{'_frozen' if is_frozen_model else ''}",
            tags=[dataset_name, model_name, "frozen" if is_frozen_model else "full"],
            config={
                "dataset": dataset_name,
                "batch_size": batch_size,

                "model": model_name,
                "model_repr": repr(model),
                "training": {
                    "optimiser": repr(optimiser),
                    "scheduler": {
                        "name": scheduler.__class__.__name__,
                        # todo: these attributes are particular to ReduceLROnPlateau
                        "reduction steps (for /10)": scheduler_reduction_steps,
                        "patience": scheduler.patience,
                        "threshold": scheduler.threshold,
                    },
                    "early_stopping_threshold": early_stop_threshold,
                },

                "wandb_init_time": time.asctime(),
                "save_path": str(weights_save_path.resolve()),
            }
        )
        lg.info(f"Initialised wandb run, id={wandb_run.id}.")
    else:
        wandb_run = None

    with tqdm(total=max_epochs, desc="Epochs") as prog_bar1:
        for epoch in range(max_epochs):
            training_loss_arr = np.zeros(0)
            training_acc_arr = np.zeros(0)

            lg.info(str(prog_bar1))
            with tqdm(total=len(training_dataloader), desc="Batches", leave=False) as prog_bar2:
                for i, data in enumerate(training_dataloader):  # type: int, dict[str, torch.Tensor]
                    images: torch.Tensor = data["image"]
                    labels: torch.Tensor = data["label"]

                    loss, acc = helpers.ml.train_step(
                        model, images, labels, loss_criterion, optimiser
                    )
                    training_loss_arr = np.append(training_loss_arr, loss)
                    training_acc_arr = np.append(training_acc_arr, acc)

                    prog_bar2.update()

                    if i > 0 and i % (len(training_dataloader) // 5) == 0:
                        training_mean_loss = training_loss_arr.mean()
                        training_mean_acc = training_acc_arr.mean()

                        prog_bar2.set_postfix(train_loss=training_mean_loss, train_acc=training_mean_acc)
                        lg.debug(str(prog_bar2))

                        if wandb_run:
                            wandb_run.log({
                                "loss/train": training_mean_loss,
                                "accuracy/train": training_mean_acc,
                                "total_steps_trained": (epoch * len(training_dataloader)) + i,
                            })

                        training_loss_arr = np.zeros(0)
                        training_acc_arr = np.zeros(0)

            val_mean_loss, val_mean_acc = helpers.ml.validation_step(
                model, loss_criterion, validation_iterator, len(validation_dataloader)
            )

            scheduler.step(val_mean_loss)
            current_lr = scheduler.get_last_lr()[0]

            prog_bar1.update()
            prog_bar1.set_postfix(val_loss=val_mean_loss, val_acc=val_mean_acc, lr=current_lr)

            if wandb_run:
                wandb_run.log({
                    "loss/validation": val_mean_loss,
                    "accuracy/validation": val_mean_acc,
                    "learning_rate": current_lr,
                })

                if epoch != 0 and epoch % 10 == 0:
                    model_save_path = weights_save_path / f"{wandb_run.id}_epoch{epoch:03}.st"
                    st.save_model(model, model_save_path)
                    lg.info(f"Saved model at epoch {epoch} to {model_save_path}.")

            if current_lr < early_stop_threshold:
                lg.info(
                    f"Early stopping on low learning rate {current_lr} "
                    f"(loss plateaued at {val_mean_loss} after lr reductions).")
                break

    model_save_path = weights_save_path / f"{wandb_run.id}_final_{val_mean_acc:.3f}.st"
    st.save_model(model, model_save_path)
    lg.info(f"Saved final model to {model_save_path}.")

    if wandb_run:
        wandb_run.summary["n_epochs"] = epoch
        wandb_run.finish(0)
        lg.info(f"Finished wandb run, id={wandb_run.id}.")


lg.info("Training partially frozen model...")
model.freeze_layers(1)  # freeze all but the last layer
if model.modified_input_layer:  # unfreeze the input layer if we need to train it too
    model.unfreeze_input_layers(model.input_layers_to_train)

try:
    train_model(lr=frozen_lr, is_frozen_model=True, max_epochs=frozen_max_epochs,
                early_stop_threshold=frozen_lr_early_stop_threshold)
except Exception as e:
    lg.exception("An exception occurred during model(frozen) training.", exc_info=e, stack_info=True)

lg.info("Training full (unfrozen) model...")
model.unfreeze_layers()

try:
    train_model(lr=full_lr, is_frozen_model=False, max_epochs=full_max_epochs,
                early_stop_threshold=lr_early_stop_threshold)
except Exception as e:
    lg.exception("An exception occurred during model training.", exc_info=e, stack_info=True)

lg.info("Script execution complete.")
