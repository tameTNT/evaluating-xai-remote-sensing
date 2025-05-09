"""
This is the main training script used to train all deep learning models
described in our paper. It should be run from the command line from the project
root directory as a module: e.g. `python -m model_training.train --help`
"""

import argparse
import json
import platform
import time
from pathlib import Path
import typing as t
import signal

import numpy as np
import safetensors.torch as st
import torch
import torch.nn as nn
# from torch.utils.viz._cycles import warn_tensor_cycles
from tqdm.autonotebook import tqdm

import dataset_processing
import helpers
import models
import wandb


SUPPORTED_OPTIMISERS = t.Literal["SGD", "Adam", "AdamW"]

# ==== Parse command line arguments ====
parser = argparse.ArgumentParser(description="Train a deep learning model on a remote sensing dataset.")

script_meta_group = parser.add_argument_group("Meta",
                                              "Arguments for script behaviour and reproducibility.")
script_meta_group.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="Specify a random PyTorch seed for reproducibility. "
         "Note that this does not affect numpy randomness. Defaults to 42.",
)
script_meta_group.add_argument(
    "--checkpoints_root_name",
    type=Path,
    default="checkpoints",
    help="Specify the directory for checkpoints with the project. Defaults to checkpoints",
)
script_meta_group.add_argument(
    "--do_not_track",
    action="store_true",
    help="If present, do not track the run using WandB.",
)
script_meta_group.add_argument(
    "--wandb_run_name",
    type=str,
    default="",
    help="Name of the WandB run."
         "If not given, the default of {dataset_name}_{model_name}{'_frozen' if is_frozen_model else ''} is used.",
)
script_meta_group.add_argument(
    "--num_workers",
    type=int,
    default=4 if platform.system() != "Windows" else 0,
    help="Number of workers to use for DataLoaders. Defaults to 4 on non-Windows systems.",
)
script_meta_group.add_argument(
    "--record_cuda_memory",
    action="store_true",
    help="Whether to record CUDA memory usage using torch.cuda.memory._record_memory_history()."
         "If the program crashes due to an OutOfMemoryError, "
         "the memory snapshot is dumped to COM_dump.pickle."
)

model_group = parser.add_argument_group("Model",
                                        "Arguments specifying the model and any weights to use.")
model_group.add_argument(
    "--model_name",
    type=str,
    required=True,
    choices=t.get_args(models.MODEL_NAMES),
    help="Name of the model to train.",
)
model_group.add_argument(
    "--use_pretrained",
    action="store_true",
    help="If given, load the model's pretrained weights.",
)
model_group.add_argument(
    "--start_from",
    type=Path,
    default="",
    help="If a path to a file is given, load a previously saved weights and start training. "
         "Best used to continue training from a previous run.",
)

dataset_group = parser.add_argument_group("Dataset",
                                          "Arguments describing the dataset to use and any pre-training processing.")
dataset_group.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    choices=t.get_args(dataset_processing.DATASET_NAMES),
    help="Name of the dataset to train on.",
)
dataset_group.add_argument(
    "--download",
    action="store_true",
    help="If given, download the dataset if it is not already present. "
         "Otherwise, script will fail if the dataset is not already downloaded.",
)
dataset_group.add_argument(
    "--normalisation_type",
    type=str,
    default="none",
    choices=["scaling", "mean_std", "none"],
    help="The type of normalisation to apply to all images in the dataset."
         "'scaling' uses min/max or percentile scaling. Note that 'mean_std' "
         "will initially calculate these values across the whole dataset the "
         "first time it is used. 'none' applies no normalisation. "
         "Note that not all datasets support all types.",
)
dataset_group.add_argument(
    "--no_resize",
    action="store_true",
    help="If given, use the original images centred with padding. "
         "Otherwise, resize and interpolate the images to the model's expected input size.",
)
dataset_group.add_argument(
    "--no_augmentations",
    action="store_true",
    help="If given, apply no random augmentations to training set images. ",
)
dataset_group.add_argument(
    "--batch_size",
    type=int,
    required=True,
    help="Batch size to use for DataLoaders.",
)

training_group = parser.add_argument_group("Training",
                                           "Arguments to specify general training options.")
training_group.add_argument(
    "--optimiser_name",
    type=str,
    default="SGD",
    choices=t.get_args(SUPPORTED_OPTIMISERS),
    help="Name of the optimiser to use. Defaults to 'SGD'.",
)
training_group.add_argument(
    "--loss_criterion_name",
    type=str,
    default="CrossEntropyLoss",
    choices=["CrossEntropyLoss"],
    help="Loss criterion to use. Defaults to 'CrossEntropyLoss'.",
)

frozen_args_group = parser.add_argument_group("Frozen model training arguments",
                                              "These arguments are only required if --use_pretrained is given "
                                              "and the model is initially (partially) frozen/being fine-tuned.")
frozen_args_group.add_argument(
    "--frozen_lr",
    type=float,
    help="Learning rate to use for training the partially frozen model.",
)
frozen_args_group.add_argument(
    "--frozen_max_epochs",
    type=int,
    help="Maximum number of epochs to train the partially frozen model.",
)
frozen_args_group.add_argument(
    "--frozen_lr_early_stop_threshold",
    type=float,
    help="Learning rate threshold for early stopping when training the partially frozen model.",
)

full_args_group = parser.add_argument_group("Full model training arguments",
                                            "These arguments are always required to train any model.")
full_args_group.add_argument(
    "--lr",
    type=float,
    required=True,
    help="Learning rate to use for training the full model.",
)
full_args_group.add_argument(
    "--max_epochs",
    type=int,
    required=True,
    help="Maximum number of epochs to train the full model.",
)
full_args_group.add_argument(
    "--lr_early_stop_threshold",
    type=float,
    required=True,
    help="Learning rate threshold for early stopping when training the full model.",
)

# Parse arguments
args: argparse.Namespace = parser.parse_args()
print("Got args:", args, "\n")

random_seed: int = args.random_seed
checkpoints_root_name: Path = args.checkpoints_root_name
do_not_track: bool = args.do_not_track
wandb_run_name: str = args.wandb_run_name
num_workers: int = args.num_workers
record_cuda_memory: bool = args.record_cuda_memory

model_name: models.MODEL_NAMES = args.model_name
use_pretrained: bool = args.use_pretrained
start_from: Path = args.start_from.expanduser().resolve()

dataset_name: dataset_processing.DATASET_NAMES = args.dataset_name
download: bool = args.download
normalisation_type: str = args.normalisation_type
use_resize: bool = not args.no_resize
use_augmentations: bool = not args.no_augmentations
batch_size: int = args.batch_size

optimiser_name: SUPPORTED_OPTIMISERS = args.optimiser_name
loss_criterion: nn.Module = getattr(nn, args.loss_criterion_name)()

frozen_lr: float = args.frozen_lr
frozen_lr_early_stop_threshold: float = args.frozen_lr_early_stop_threshold
frozen_max_epochs: int = args.frozen_max_epochs

lr: float = args.lr
lr_early_stop_threshold: float = args.lr_early_stop_threshold
max_epochs: int = args.max_epochs


# ==== Actual training script starts here ====
logger = helpers.log.main_logger
# noinspection PyUnresolvedReferences
print(f"Logging to {logger.handlers[0].baseFilename}. See file for details.\n")
logger.debug(f"Running script with args: {args}")


# Adapted from https://stackoverflow.com/a/31464349/7253717
# Allows for graceful shutdown of the script (hopefully)
class GracefulKiller:
    please_kill = False

    def __init__(self):
        # SLURM will send a signal to the script to kill it
        signal.signal(signal.SIGINT, self.exit_gracefully)   # interrupt signal
        signal.signal(signal.SIGTERM, self.exit_gracefully)  # termination signal
        # signal.signal(signal.SIGKILL, self.exit_gracefully)  # you can't catch SIGKILL

    def exit_gracefully(self, signum, frame):
        print("Interrupt/termination signal received!")
        self.please_kill = True


killer = GracefulKiller()


if platform.system() == "Windows":
    num_workers = 0
    logger.warning("num_workers != 0 is not supported on Windows. Setting num_workers=0.")

torch_device = helpers.utils.get_torch_device()

if record_cuda_memory:
    Path("memory_dumps").mkdir(exist_ok=True)
    logger.info("record_cuda_memory is enabled. CUDA memory usage will be recorded and dumped "
                "to COM_dump.pickle in the event of an OutOfMemoryError.")
    torch.cuda.memory._record_memory_history()

np_rng = np.random.default_rng(random_seed)
_ = torch.manual_seed(random_seed)
logger.debug(f'Random seed set to {random_seed}.')

# ==== Loads model and training dataset ====
checkpoints_path = helpers.env_var.get_project_root() / checkpoints_root_name
checkpoints_path.mkdir(exist_ok=True)
logger.debug(f'Checkpoints directory set to {checkpoints_path}.')

model_type = models.get_model_type(model_name)
training_dataset = dataset_processing.get_dataset_object(
    dataset_name, "train", model_type.expected_input_dim,
    normalisation_type=normalisation_type, use_augmentations=use_augmentations, use_resize=use_resize,
    batch_size=batch_size, num_workers=num_workers, device=torch_device,
    download=download,
)

validation_dataset = dataset_processing.get_dataset_object(
    dataset_name, "val", model_type.expected_input_dim,
    normalisation_type=normalisation_type, use_resize=use_resize,
    batch_size=batch_size, num_workers=num_workers, device=torch_device,
    download=download,
)

model = model_type(
    pretrained=use_pretrained,
    n_input_bands=training_dataset.N_BANDS,
    n_output_classes=training_dataset.N_CLASSES
).to(torch_device)

multiprocessing_context = None
training_dataloader = torch.utils.data.DataLoader(
    training_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True,
    multiprocessing_context=multiprocessing_context, pin_memory=True,
)
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    multiprocessing_context=multiprocessing_context, pin_memory=True,
)
sampling_dataloader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True,
    multiprocessing_context=multiprocessing_context, pin_memory=True,
)

validation_iterator = iter(dataset_processing.core.cycle(validation_dataloader))
sampling_iterator = iter(dataset_processing.core.cycle(sampling_dataloader))


def get_opt_and_scheduler(opt_lr: float, reduction_steps: int = 4):
    opt_kwargs = {}
    if optimiser_name == "SGD":
        opt_kwargs = {
            "weight_decay": 1e-6, "momentum": 0.9, "nesterov": True,
        }
    elif optimiser_name == "AdamW":
        opt_kwargs = {
            "weight_decay": 1e-2, "betas": (0.9, 0.999), "eps": 1e-8,
        }
    opt: torch.optim.Optimizer = getattr(torch.optim, optimiser_name)(
        filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr, **opt_kwargs
    )

    # futuretodo: add support for other schedulers
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=np.float_power(10, -1 / reduction_steps),
        # requires reduction_steps reductions to reduce by factor 10 (*0.1)
        patience=5, threshold=0.1  # futuretodo: add these as script args
    )
    return opt, sch


# ==== Main train function called below ====
# futuretodo: support full loading and resuming training from a previous saved state
def train_model(
        train_lr: float,
        is_frozen_model: bool = False,
        train_max_epochs: int = 50,
        scheduler_reduction_steps: int = 4,  # futuretodo: add as script arg
        early_stop_threshold: float = 0.0001
):
    weights_save_path = checkpoints_path / training_dataset.__class__.__name__ / model.__class__.__name__
    if is_frozen_model:
        weights_save_path /= "frozen_partial"
    else:
        weights_save_path /= "full"
    weights_save_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output path set to {weights_save_path.resolve()}.")

    optimiser, scheduler = get_opt_and_scheduler(train_lr, scheduler_reduction_steps)
    logger.debug(f"Initialised optimiser (lr={train_lr}) and scheduler.")

    if not do_not_track:
        name_str = f"{dataset_name}_{model_name}{'_frozen' if is_frozen_model else ''}"
        wandb_run = wandb.init(
            save_code=True,
            project="evaluating_xAI_for_RS",
            name=wandb_run_name if wandb_run_name != "" else name_str,
            tags=[dataset_name, model_name, "frozen" if is_frozen_model else "full"],
            config={
                "full_cli_args": repr(args),
                "dataset": dataset_name,
                "transform_options": {
                    "normalisation_type": normalisation_type,
                    "use_augmentations": use_augmentations,
                    "use_resize": use_resize,
                    "transforms": training_dataset.composed_transforms,
                },
                "batch_size": batch_size,

                "model": model_name,
                "model_repr": repr(model),
                "training": {
                    "optimiser": repr(optimiser),
                    "scheduler": {
                        "name": scheduler.__class__.__name__,
                        # futuretodo: these attributes are particular to ReduceLROnPlateau - generalise
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
        logger.info(f"Initialised wandb run, id={wandb_run.id}.")
    else:
        class DummyRun:
            def __init__(self, id_: str):
                self.id = id_

            def __bool__(self):
                return False  # this is a fake run so evaluates to False in if statements

        wandb_run = DummyRun(f"untracked_{int(time.time())}")

    # warn_tensor_cycles()
    val_mean_acc = 0.0
    with tqdm(total=train_max_epochs, unit="epoch", ncols=110) as prog_bar1:
        for epoch in range(train_max_epochs):
            training_loss_arr = np.zeros(0)
            training_acc_arr = np.zeros(0)

            logger.info(str(prog_bar1))
            # == Main training loop ==
            with tqdm(total=len(training_dataloader), desc="Training",
                      unit="batch", ncols=110, leave=False) as prog_bar2:
                for i, data in enumerate(training_dataloader):  # type: int, dict[str, torch.Tensor]
                    if killer.please_kill:
                        model_save_path = weights_save_path / (f"{wandb_run.id}_"
                                                               f"killed_at_{epoch:03}_{val_mean_acc:.3f}.st")
                        logger.warning(f"Received termination/kill signal. "
                                       f"Saving model to {model_save_path} and exiting.")
                        st.save_model(model, model_save_path)
                        raise KeyboardInterrupt("Received kill signal (from SLURM).")

                    images: torch.Tensor = data["image"]
                    labels: torch.Tensor = data["label"]

                    loss, acc = helpers.ml.train_step(
                        model, images, labels, loss_criterion, optimiser
                    )
                    training_loss_arr = np.append(training_loss_arr, loss)
                    training_acc_arr = np.append(training_acc_arr, acc)

                    prog_bar2.update()

                    if i == 2 and record_cuda_memory:
                        logger.debug("Dumping CUDA memory usage to first_3_iterations.pickle.")
                        torch.cuda.memory._dump_snapshot("memory_dumps/first_3_iterations.pickle")
                        torch.cuda.memory._record_memory_history(enabled=None)
                    if i == len(training_dataloader) - 2 and record_cuda_memory:
                        logger.debug("Re-enabling CUDA memory recording 1 iteration before validation.")
                        torch.cuda.memory._record_memory_history()

                    if i > 0 and i % (len(training_dataloader) // 5) == 0:
                        training_mean_loss = training_loss_arr.mean()
                        training_mean_acc = training_acc_arr.mean()

                        prog_bar2.set_postfix(train_loss=training_mean_loss, train_acc=training_mean_acc)
                        logger.debug(str(prog_bar2))

                        # == Log metrics for this training step ==
                        if wandb_run:
                            wandb_run.log({
                                "loss/train": training_mean_loss,
                                "accuracy/train": training_mean_acc,
                                "total_steps_trained": (epoch * len(training_dataloader)) + i,
                            })

                        training_loss_arr = np.zeros(0)
                        training_acc_arr = np.zeros(0)

            # == Post training step evaluation ==
            val_mean_loss, val_mean_acc = helpers.ml.validation_step(
                model, loss_criterion, validation_iterator, len(validation_dataloader)
            )

            previous_lr = scheduler.get_last_lr()[0]
            scheduler.step(val_mean_loss)
            current_lr = scheduler.get_last_lr()[0]
            if previous_lr != current_lr:
                logger.info(f"Learning rate updated via scheduler to {previous_lr}->{current_lr}.")

            prog_bar1.update()
            prog_bar1.set_postfix(val_loss=val_mean_loss, val_acc=val_mean_acc, lr=current_lr)

            # Save the model to file as we go along (overwritten every epoch - just as a backup to resume training)
            model_save_path = weights_save_path / f"{wandb_run.id}_current.st"
            st.save_model(model, model_save_path)
            json.dump(
                {"completed_epoch": epoch, "lr": current_lr, "args": str(args)},
                model_save_path.with_suffix(".json").open("w")
            )
            logger.debug(f"Saved current model and state at epoch {epoch} to {model_save_path}.")

            # == Log metrics for this validation step (includes some incorrect samples) ==
            if wandb_run:
                logger.info("Sampling incorrect predictions for logging to WandB...")
                samples, samples_labels, sample_outputs = helpers.ml.sample_outputs(
                    model, sampling_iterator, 1
                )
                predicted_labels = sample_outputs.argmax(dim=1)
                incorrect_mask = predicted_labels != samples_labels
                incorrect_samples = samples[incorrect_mask]
                if len(incorrect_samples) > 0:  # at least 1 incorrect sample
                    scaled_incorrect_samples = validation_dataset.inverse_transform(incorrect_samples)
                    labels_of_incorrect = samples_labels[incorrect_mask]
                    incorrect_preds = predicted_labels[incorrect_mask]

                    samples_table = wandb.Table(columns=[])
                    samples_table.add_column("epoch",
                                             (torch.ones_like(incorrect_preds) * epoch).numpy())
                    samples_table.add_column("prediction",
                                             [validation_dataset.classes[idx] for idx in incorrect_preds])
                    samples_table.add_column("label",
                                             [validation_dataset.classes[idx] for idx in labels_of_incorrect])
                    samples_table.add_column("true_sample",
                                             [wandb.Image(img) for img in scaled_incorrect_samples])
                    wandb_run.log({
                        "samples/incorrect": samples_table,
                    }, commit=False)  # don't commit this step yet

                wandb_run.log({
                    "loss/validation": val_mean_loss,
                    "accuracy/validation": val_mean_acc,
                    "learning_rate": current_lr,
                })

            if epoch != 0 and epoch % 10 == 0:
                model_save_path = weights_save_path / f"{wandb_run.id}_epoch{epoch:03}.st"
                st.save_model(model, model_save_path)
                logger.info(f"Saved model at epoch {epoch} to {model_save_path}.")

            if current_lr < early_stop_threshold:
                logger.info(
                    f"Early stopping on low learning rate {current_lr} "
                    f"(loss plateaued at {val_mean_loss} after lr reductions).")
                break

            if record_cuda_memory:
                logger.debug("Dumping CUDA memory usage to epoch_end.pickle.")
                torch.cuda.memory._dump_snapshot("memory_dumps/epoch_end.pickle")

    model_save_path = weights_save_path / f"{wandb_run.id}_final_{val_mean_acc:.3f}.st"
    st.save_model(model, model_save_path)
    logger.info(f"Saved final model to {model_save_path}.")

    if wandb_run:
        wandb_run.summary["n_epochs"] = epoch
        wandb_run.finish(0)
        logger.info(f"Finished wandb run, id={wandb_run.id}.")


def cuda_memory_dump(exception: Exception, is_frozen: bool):
    logger.exception(
        f"An OutOfMemoryError occurred during model{'(frozen)' if is_frozen else ''} training. "
        f"record_cuda_memory={record_cuda_memory} so "
        f"{'a dump was written to COM_dump.pickle' if record_cuda_memory else 'no dump was written.'}",
        exc_info=exception, stack_info=True
    )
    if record_cuda_memory:
        torch.cuda.memory._dump_snapshot("memory_dumps/COM_dump.pickle")

    raise exception


# Load and resume training from existing weights if specified.
if start_from.is_file() and start_from.suffix in (".st", ".safetensors"):
    logger.info(f"Loading weights from {start_from}...")
    try:
        st.load_model(model, start_from, device=str(torch_device), strict=True)
    except RuntimeError as error:
        loggable_error = str(error).replace("\n", " ")
        logger.warning(f"Could not load weights from {start_from} via safetensors.load_model: {loggable_error}")

# ==== Primary training loops initiated here ====
# == Optional fine-tuning step if using pretrained weights ==
if use_pretrained and frozen_lr and frozen_max_epochs and frozen_lr_early_stop_threshold:
    logger.info("Training partially frozen pretrained model...")
    model.freeze_layers(1)  # freeze all but the last linear layer
    if model.modified_input_layer:  # unfreeze the input layer if we need to train it too
        model.unfreeze_input_layers(model.input_layers_to_train)

    try:
        train_model(train_lr=frozen_lr, is_frozen_model=True, train_max_epochs=frozen_max_epochs,
                    early_stop_threshold=frozen_lr_early_stop_threshold)
    except torch.OutOfMemoryError as e:
        cuda_memory_dump(e, True)
    except Exception as e:
        logger.exception("An exception occurred during model(frozen) training.", exc_info=e, stack_info=True)
        raise e

elif use_pretrained:
    logger.warning("--use_pretrained was given but no frozen training parameters were provided. "
                   "The model will not be fine-tuned while partially frozen and only fully.")
elif frozen_lr or frozen_max_epochs or frozen_lr_early_stop_threshold:
    logger.warning("--frozen_lr, --frozen_max_epochs and/or --frozen_lr_early_stop_threshold were given but "
                   "--use_pretrained was not. Ignoring these arguments.")

# == Train the full model ==
logger.info("Training full (unfrozen) model...")
model.unfreeze_all_layers()

try:
    train_model(train_lr=lr, is_frozen_model=False, train_max_epochs=max_epochs,
                early_stop_threshold=lr_early_stop_threshold)
except torch.OutOfMemoryError as e:
    cuda_memory_dump(e, False)
except Exception as e:
    logger.exception("An exception occurred during model training.", exc_info=e, stack_info=True)
    raise e

logger.info("Script execution complete.")
