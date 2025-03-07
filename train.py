import argparse
import platform
import time
import typing as t
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

import dataset_processing
import helpers
import wandb

# Create argument parser
parser = argparse.ArgumentParser(description="Train a model on a land use dataset.")

# General arguments
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="Specify a random PyTorch seed for reproducibility. "
         "Note that this does not affect numpy randomness. Defaults to 42.",
)
parser.add_argument(
    "--checkpoints_root_name",
    type=Path,
    default="checkpoints",
    help="Specify the directory for checkpoints with l3_project. Defaults to ~/checkpoints",
)
parser.add_argument(
    "--do_not_track",
    action="store_true",
    help="If present, do not track the run using WandB.",
)

# Model arguments
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    choices=helpers.models.MODEL_NAMES,
    help="Name of the model to train.",
)
parser.add_argument(
    "--use_pretrained",
    action="store_true",
    help="If given, load the model's pretrained weights.",
)

# Dataset arguments
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    choices=dataset_processing.core.DATASET_NAMES,
    help="Name of the dataset to train on.",
)
parser.add_argument(
    "--no_resize",
    action="store_true",
    help="If given, use the original images centred with padding. "
         "Otherwise, resize and interpolate the images to the model's expected input size.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    required=True,
    help="Batch size to use for DataLoaders.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="Number of workers to use for DataLoaders. Defaults to 4.",
)

# Optimiser and loss criterion arguments
parser.add_argument(
    "--optimiser_name",
    type=str,
    default="SGD",
    choices=["SGD", "Adam"],  # todo: support more optimisers
    help="Name of the optimiser to use. Defaults to SGD.",
)
parser.add_argument(
    "--loss_criterion_name",
    type=str,
    default="CrossEntropyLoss",
    choices=["CrossEntropyLoss"],  # todo: support more loss criteria
    help="Loss criterion to use. Defaults to CrossEntropyLoss.",
)

# Frozen model training arguments
parser.add_argument(
    "--frozen_lr",
    type=float,
    default=0.01,
    help="(If using a pretrained model) Learning rate to use for training the partially frozen model.",
)
parser.add_argument(
    "--frozen_max_epochs",
    type=int,
    default=20,
    help="(If using a pretrained model) Maximum number of epochs to train the partially frozen model.",
)
parser.add_argument(
    "--frozen_lr_early_stop_threshold",
    type=float,
    default=0.0001,
    help="(If using a pretrained model) Learning rate threshold for early stopping when training frozen model.",
)

# Full model training arguments
parser.add_argument(
    "--full_lr",
    type=float,
    required=True,
    help="Learning rate to use for training the full model.",
)
parser.add_argument(
    "--full_max_epochs",
    type=int,
    required=True,
    help="Maximum number of epochs to train the full model.",
)
parser.add_argument(
    "--lr_early_stop_threshold",
    type=float,
    required=True,
    help="Learning rate threshold for early stopping when training full model.",
)

# Parse arguments
args = parser.parse_args()
print("Got args:", args, "\n")

random_seed = args.random_seed
checkpoints_root_name = args.checkpoints_root_name
do_not_track = args.do_not_track

model_name = args.model_name
use_pretrained = args.use_pretrained

dataset_name = args.dataset_name
use_resize = not args.no_resize
batch_size = args.batch_size
num_workers = args.num_workers

optimiser_name = args.optimiser_name
loss_criterion: nn.Module = getattr(nn, args.loss_criterion_name)()

frozen_lr = args.frozen_lr
frozen_lr_early_stop_threshold = args.frozen_lr_early_stop_threshold
frozen_max_epochs = args.frozen_max_epochs

full_lr = args.full_lr
lr_early_stop_threshold = args.lr_early_stop_threshold
full_max_epochs = args.full_max_epochs


# Actual script starts here
lg = helpers.logging.get_logger("main")
print(f"Logging to {lg.handlers[0].baseFilename}. See file for details.\n")
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
        name: str,
        split: t.Literal["train", "val", "test"],
        image_size: int,
        **kwargs
) -> t.Union[dataset_processing.eurosat.EuroSATBase]:
    standard_kwargs = {
        "split": split,
        "image_size": image_size,
    }

    if name == "EuroSATRGB":
        lg.debug("Loading EuroSATRGB dataset...")
        ds = dataset_processing.eurosat.EuroSATRGB(**standard_kwargs, **kwargs)
    elif name == "EuroSATMS":
        lg.debug("Loading EuroSATMS dataset...")
        ds = dataset_processing.eurosat.EuroSATMS(**standard_kwargs, **kwargs)
    else:
        lg.error(f"Invalid dataset name ({name}) provided to get_dataset_object.")
        raise ValueError(f"Dataset {name} does not exist.")

    lg.info(f"Dataset {name} ({split}) loaded with {len(ds)} samples.")
    return ds


def get_model_type(
        name: str,
) -> t.Type[helpers.models.FreezableModel]:
    if name == "ResNet50":
        lg.debug("Returning ResNet50 model type...")
        m = helpers.models.FineTunedResNet50
    else:
        lg.error(f"Invalid model name ({name}) provided to get_model_type.")
        raise ValueError(f"Model {name} does not exist.")

    return m


model_type = get_model_type(model_name)

# manual_mean_std_calc = [[1118.3116455078125, 1043.06982421875, 947.53662109375, 1199.5548095703125, 1999.6763916015625,
#                          2368.963134765625, 2296.98486328125, 732.1416015625, 2594.695068359375],
#                         [327.0133056640625, 388.61962890625, 586.3471069335938, 565.2821655273438, 859.8197631835938,
#                          1084.8768310546875, 1107.9259033203125, 404.88214111328125, 1229.2401123046875]]
training_dataset = get_dataset_object(
    dataset_name, "train", model_type.expected_input_dim, normalisation_type="scaling",
    use_augmentations=True, use_resize=use_resize, batch_size=batch_size,
)
validation_dataset = get_dataset_object(
    dataset_name, "val", model_type.expected_input_dim, normalisation_type="scaling",
    use_augmentations=True, use_resize=use_resize, batch_size=batch_size,
)

model = model_type(
    pretrained=use_pretrained,
    n_input_bands=training_dataset.N_BANDS,
    n_output_classes=training_dataset.N_CLASSES
).to(torch_device)

training_dataloader = torch.utils.data.DataLoader(
    training_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True
)
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
)
sampling_dataloader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True
)

validation_iterator = iter(dataset_processing.core.cycle(validation_dataloader))
sampling_iterator = iter(dataset_processing.core.cycle(sampling_dataloader))


def get_opt_and_scheduler(lr: float, reduction_steps: int = 4):
    kwargs = {}
    if optimiser_name == "SGD":
        kwargs = {
            "weight_decay": 1e-6, "momentum": 0.9, "nesterov": True
        }
    opt: torch.optim.Optimizer = getattr(torch.optim, optimiser_name)(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, **kwargs
    )

    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=np.float_power(10, -1 / reduction_steps),
        # requires reduction_steps reductions to reduce by factor 10 (*0.1)
        patience=5, threshold=0.1  # todo: add these as args
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

    if not do_not_track:
        wandb_run = wandb.init(
            save_code=True,
            project="evaluating_xAI_for_RS",
            name=f"{dataset_name}_{model_name}{'_frozen' if is_frozen_model else ''}",
            tags=[dataset_name, model_name, "frozen" if is_frozen_model else "full"],
            config={
                "dataset": dataset_name,
                "transform_options": {
                    "use_normalisation": True,  # todo: make this a script arg
                    "use_augmentations": True,  # todo: make this an explicit list
                    "use_resize": use_resize,
                },
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

    with tqdm(total=max_epochs, unit="epoch", ncols=110) as prog_bar1:
        for epoch in range(max_epochs):
            training_loss_arr = np.zeros(0)
            training_acc_arr = np.zeros(0)

            lg.info(str(prog_bar1))
            with tqdm(total=len(training_dataloader), desc="Training",
                      unit="batch", ncols=110, leave=False) as prog_bar2:
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
                lg.info("Sampling incorrect predictions for logging to WandB...")
                samples, samples_labels, sample_outputs = helpers.ml.sample_outputs(
                    model, sampling_iterator, 1
                )
                predicted_labels = sample_outputs.argmax(dim=1)
                incorrect_mask = predicted_labels != samples_labels
                incorrect_samples = samples[incorrect_mask]
                # select RGB bands (for training on eurosat MS) and rescale todo: automate inverse transform
                scaled_incorrect_samples = ((incorrect_samples[:, [2, 1, 0]] + 1) / 2).clamp(0, 1)
                labels_for_incorrect = samples_labels[incorrect_mask]
                incorrect_preds = predicted_labels[incorrect_mask]

                samples_table = wandb.Table(columns=[])
                samples_table.add_column("epoch", (torch.ones_like(incorrect_preds) * epoch).numpy())
                samples_table.add_column("prediction", incorrect_preds.numpy())
                samples_table.add_column("label", labels_for_incorrect.numpy())
                samples_table.add_column("true_sample", [wandb.Image(
                    img.numpy().transpose(1, 2, 0), caption=validation_dataset.classes[int(label)]
                ) for img, label in zip(scaled_incorrect_samples, labels_for_incorrect)])

                wandb_run.log({
                    "loss/validation": val_mean_loss,
                    "accuracy/validation": val_mean_acc,
                    "learning_rate": current_lr,
                    "samples/incorrect": samples_table,
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


if use_pretrained:
    lg.info("Training partially frozen pretrained model...")
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
