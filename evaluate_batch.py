import argparse
import platform
import typing as t
from pathlib import Path
import json
import functools

import torch
import safetensors.torch as st
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import models
import dataset_processing
import xai
import evaluate_xai.deletion
from evaluate_xai.correctness import Correctness
from evaluate_xai.output_completeness import OutputCompleteness
from evaluate_xai.continuity import Continuity
from evaluate_xai.contrastivity import Contrastivity
from evaluate_xai.compactness import Compactness
import helpers

torch_device = helpers.utils.get_torch_device()
logger = helpers.log.main_logger

SHAP_MAX_EVALS = 10000


def get_data_and_model() -> tuple:
    logger.debug("Loading dataset and model...")

    model_type = models.get_model_type(model_name)
    ds = dataset_processing.get_dataset_object(
        dataset_name, "val", model_type.expected_input_dim,  # todo: switch to test in production
        normalisation_type=normalisation_type, use_resize=True,
        batch_size=batch_size, num_workers=num_workers, device=torch_device,
        download=False,
    )

    weights_path = json.load(Path("weights_paths.json").open("r"))[dataset_name][model_name]
    model_weights_path = checkpoints_path / dataset_name / model_name / weights_path

    logger.debug(f"Loading pretrained weights from {model_weights_path} and loading into model...")
    model = model_type(
        pretrained=False, n_input_bands=ds.N_BANDS, n_output_classes=ds.N_CLASSES,
    )
    st.load_model(model, model_weights_path)
    model.eval().to(torch_device)

    return ds, model


def get_explainer_args() -> dict:
    explain_args = {}
    if explainer_name == "PartitionSHAP":
        explain_args["batch_size"] = batch_size
        explain_args["max_evals"] = SHAP_MAX_EVALS

    return explain_args


def generate_explanations(_for_idxs: np.array, class_idx: int) -> list[xai.Explainer]:
    num_batches = int(np.ceil(len(_for_idxs) / batch_size))
    explainers = []
    for i in tqdm(range(num_batches), unit="batch", ncols=110, leave=False):
        idxs_for_batch = _for_idxs[i*batch_size:(i+1)*batch_size]
        batch = torch.stack([dataset[j]["image"] for j in idxs_for_batch])

        explainer = xai.get_explainer_object(
            explainer_name, model_to_explain,
            extra_path=Path(dataset_name) / f"c{class_idx:02}" / f"b{i:03}",
        )
        if not explainer.has_explanation_for(batch):
            logger.debug(f"No existing explanation for batch {i} of class {class_idx}. Generating new ones.")
            explainer.explain(batch, **get_explainer_args())
        explainers.append(explainer)

    return explainers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an xAI method on a selected deep learning model and dataset."
    )

    script_meta_group = parser.add_argument_group("Meta",
                                                  "Arguments for script behaviour and reproducibility.")
    script_meta_group.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Specify a random PyTorch/NumPy seed for reproducibility. Defaults to 42.",
    )
    script_meta_group.add_argument(
        "--checkpoints_path",
        type=Path,
        default="checkpoints",
        help="Specify the directory for checkpoints with the project. Defaults to './checkpoints/'",
    )
    script_meta_group.add_argument(
        "--num_workers",
        type=int,
        default=4 if platform.system() != "Windows" else 0,
        help="Number of workers to use for DataLoaders. Defaults to 4 on non-Windows systems.",
    )
    script_meta_group.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size to use for DataLoaders and xAI methods."
             "Try reducing this if there are any memory errors.",
    )

    options_group = parser.add_argument_group("Primary Options",
                                              "Specify key options for the script. All required.")

    options_group.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=t.get_args(dataset_processing.DATASET_NAMES),
        help="Name of the dataset to evaluate the model on.",
    )
    options_group.add_argument(
        "--normalisation_type",
        type=str,
        default="none",
        choices=["scaling", "mean_std", "none"],
        help="The type of normalisation to apply to all images in the dataset."
             "'scaling' uses min/max or percentile scaling. Note that 'mean_std' "
             "will initially calculate these values across the whole dataset the "
             "first time it is used. 'none' applies no normalisation. "
             "Only required for the EuroSAT datasets. Defaults to 'none'.",
    )
    options_group.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=t.get_args(models.MODEL_NAMES),
        help="Name of the model to evaluate.",
    )
    options_group.add_argument(
        "--explainer_name",
        type=str,
        required=True,
        choices=t.get_args(xai.EXPLAINER_NAMES),
        help="Name of the xAI method to evaluate.",
    )

    evaluation_group = parser.add_argument_group("Evaluation Options",
                                                 "Options for the evaluation of the xAI method.")
    evaluation_group.add_argument(
        "--samples_per_class",
        type=int,
        default=0,
        help="Number of samples to evaluate per class. If not given or <=0, "
             "generates explanations for all samples in the class.",
    )
    evaluation_group.add_argument(
        "--deletion_method",
        type=str,
        default="shuffle",
        choices=evaluate_xai.deletion.METHODS,
        help="Method to use for deletion/perturbation-related evaluation methods. "
             "Defaults to 'shuffle'.",
    )
    evaluation_group.add_argument(
        "--similarity_intersection_k",
        type=int,
        default=5000,
        help="Number of pixels to use for the intersection similarity metric. "
             "Defaults to 5000.",
    )
    evaluation_group.add_argument(
        "--deletion_iterations",
        type=int,
        default=15,
        help="Number of iterations to use for the incremental methods. "
             "Defaults to 15.",
    )
    evaluation_group.add_argument(
        "--num_random_trials",
        type=int,
        default=5,
        help="Number of random trials to use for the methods with a random comparison. "
             "Defaults to 5.",
    )

    args: argparse.Namespace = parser.parse_args()
    print("Got args:", args, "\n")

    random_seed: int = args.random_seed
    checkpoints_path: Path = args.checkpoints_path.expanduser().resolve()
    num_workers: int = args.num_workers
    batch_size: int = args.batch_size

    dataset_name: dataset_processing.DATASET_NAMES = args.dataset_name
    normalisation_type: str = args.normalisation_type
    model_name: models.MODEL_NAMES = args.model_name
    explainer_name: xai.EXPLAINER_NAMES = args.explainer_name

    samples_per_class: int = args.samples_per_class
    deletion_method: evaluate_xai.deletion.METHODS = args.deletion_method
    similarity_intersection_k: int = args.similarity_intersection_k
    deletion_iterations: int = args.deletion_iterations
    num_random_trials: int = args.num_random_trials
    
    logger.info(f"Successfully got script arguments: {args}.")

    torch.manual_seed(random_seed)
    np_rng = np.random.default_rng(random_seed)

    dataset, model_to_explain = get_data_and_model()

    results_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
        [("correctness", "model_randomisation"), ("correctness", "incremental_deletion"),
         ("output_completeness", "deletion_check"), ("output_completeness", "preservation_check"),
         ("continuity", "perturbation"), ("contrastivity", "adversarial_attack"),
         ("compactness", "threshold")]
    ), index=range(dataset.N_CLASSES))

    classes = np.array([class_ for _, class_ in dataset.imgs])
    for c in tqdm(range(dataset.N_CLASSES), ncols=110, desc="xAI per class"):
        class_idxs = np.where(classes == c)[0]

        num_samples_to_take = samples_per_class
        if len(class_idxs) < num_samples_to_take:
            logger.warning(f"samples_per_class={samples_per_class} > num samples for class {c} ({len(class_idxs)}). "
                           f"Using ({len(class_idxs)}).")
            num_samples_to_take = len(class_idxs)
        elif samples_per_class <= 0:
            num_samples_to_take = len(class_idxs)
        class_idxs_sampled = np_rng.choice(class_idxs, num_samples_to_take, replace=False)

        explainers_for_c = generate_explanations(class_idxs_sampled, c)
        # Combine all the explainers for this class into one
        combined_exp = functools.reduce(lambda x, y: x | y, explainers_for_c)

        # helpers.plotting.visualise_importance(combined_exp.input, combined_exp.ranked_explanation,
        #                                       alpha=.2, with_colorbar=False)
        # plt.title(f"Explanations for Class {c:02}")
        # plt.show()

        # ==== Evaluate Correctness ====
        correctness = Correctness(combined_exp, max_batch_size=batch_size)

        correctness_mr_dict = correctness.evaluate(
            method="model_randomisation", visualise=False,
        )(l2_normalise=True, intersection_k=similarity_intersection_k)
        top_k_intersection_with_randomised: np.ndarray = correctness_mr_dict["top_k_intersection"]

        correctness_id_dict = correctness.evaluate(
            method="incremental_deletion",
            deletion_method=deletion_method,
            iterations=deletion_iterations, n_random_rankings=num_random_trials,
            random_seed=random_seed, visualise=False,
        )

        # ==== Save all results in the dataframe ====
        results_df.loc[c] = [
            # We want the intersection to be low since the model was randomised
            top_k_intersection_with_randomised.mean(),
            # We expect the informed deletion to be smaller than the randomised one so this should be small
            (correctness_id_dict["informed"]/correctness_id_dict["random"]).mean(),
            # todo: include remaining metrics
        ]

    store = pd.HDFStore(str(Path("output.csv")))
    store["results_df"] = results_df  # saves results_df object to HDF5 file

else:
    raise RuntimeError("Please run this script from the command line.")
