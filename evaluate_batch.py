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
from evaluate_xai import Similarity
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
            attempt_load=batch,
        )
        if not explainer.has_explanation_for(batch):
            logger.debug(f"No existing explanation for batch {i} of class {class_idx}. Generating new ones.")
            explainer.explain(batch, **get_explainer_args())
        explainers.append(explainer)

    return explainers


def evaluate_sim_to_array(sim: Similarity) -> np.array:
    sim_return_dict = sim(l2_normalise=True, intersection_k=similarity_intersection_k)
    return np.stack([sim_return_dict[m] for m in available_sim_metrics])


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
    script_meta_group.add_argument(
        "--visualise",
        action="store_true",
        help="If set, visualise the explanations generated and evaluations performed.",
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
    metric_options = parser.add_argument_group("Metric Options",
                                               "Options for specific evaluation metrics.")
    metric_options.add_argument(
        "--output_completeness_threshold",
        type=float,
        default=0.2,
        help="Proportion of images to delete/preserve for output completeness evaluation metrics. "
             "Defaults to 0.2.",
    )
    metric_options.add_argument(
        "--continuity_perturbation_degree",
        type=float,
        default=0.15,
        help="Severity of perturbation to apply for continuity evaluation. "
             "Defaults to 0.15.",
    )
    metric_options.add_argument(
        "--compactness_threshold",
        type=float,
        default=0.5,
        help="Threshold to use for compactness evaluation. "
             "Defaults to 0.5.",
    )

    shared_options = parser.add_argument_group("Shared Options",
                                               "General shared options for xAI evaluation.")
    shared_options.add_argument(
        "--samples_per_class",
        type=int,
        default=0,
        help="Number of samples to evaluate per class. If not given or <=0, "
             "generates explanations for all samples in the class.",
    )
    shared_options.add_argument(
        "--deletion_method",
        type=str,
        default="shuffle",
        choices=evaluate_xai.deletion.METHODS,
        help="Method to use for deletion/perturbation-related evaluation methods. "
             "Defaults to 'shuffle'.",
    )
    shared_options.add_argument(
        "--deletion_iterations",
        type=int,
        default=15,
        help="Number of iterations to use for the incremental methods. "
             "Defaults to 15.",
    )
    shared_options.add_argument(
        "--num_random_trials",
        type=int,
        default=5,
        help="Number of random trials to use for the methods with a random comparison element. "
             "Defaults to 5.",
    )
    shared_options.add_argument(
        "--similarity_intersection_proportion",
        type=float,
        default=0.1,
        help="Proportional of image pixels to use for the intersection similarity metric. "
             "Defaults to 10% (0.1).",
    )

    args: argparse.Namespace = parser.parse_args()
    print("Got args:", args, "\n")

    random_seed: int = args.random_seed
    checkpoints_path: Path = args.checkpoints_path.expanduser().resolve()
    num_workers: int = args.num_workers
    batch_size: int = args.batch_size
    visualise: bool = args.visualise

    dataset_name: dataset_processing.DATASET_NAMES = args.dataset_name
    normalisation_type: str = args.normalisation_type
    model_name: models.MODEL_NAMES = args.model_name
    explainer_name: xai.EXPLAINER_NAMES = args.explainer_name

    output_completeness_threshold: float = args.output_completeness_threshold
    continuity_perturbation_degree: float = args.continuity_perturbation_degree
    compactness_threshold: float = args.compactness_threshold

    samples_per_class: int = args.samples_per_class
    deletion_method: evaluate_xai.deletion.METHODS = args.deletion_method
    deletion_iterations: int = args.deletion_iterations
    num_random_trials: int = args.num_random_trials
    similarity_intersection_proportion: float = args.similarity_intersection_proportion

    print(f"Logging to {logger.handlers[0].baseFilename}. See file for details.\n")
    logger.info(f"Successfully got script arguments: {args}.")

    torch.manual_seed(random_seed)
    np_rng = np.random.default_rng(random_seed)

    dataset, model_to_explain = get_data_and_model()

    similarity_intersection_k = int(similarity_intersection_proportion * model_to_explain.expected_input_dim ** 2)
    available_sim_metrics = t.get_args(evaluate_xai.SIMILARITY_METRICS)

    results_df = pd.DataFrame(columns=[
        # ↓, low similarity
        *[f"correctness : randomised_model_similarity : {metric_name}" for metric_name in available_sim_metrics],
        # ↓, best is 0
        "correctness : incremental_deletion_auc_ratio",
        # ↑, best is 1
        "output_completeness : deletion_check_conf_drop",
        # ↓, best is 0
        "output_completeness : preservation_check_conf_drop",
        # ↑, high similarity
        *[f"continuity : perturbation_similarity : {metric_name}" for metric_name in available_sim_metrics],
        # ↓, low similarity
        *[f"contrastivity : adversarial_attack_similarity : {metric_name}" for metric_name in available_sim_metrics],
        # ↑, near 1
        "compactness : threshold_score"
    ], index=dataset.classes)  # new row for each dataset class

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

        if visualise:
            # todo: stack large numbers like with visualise_incremental_deletion
            helpers.plotting.visualise_importance(combined_exp.input, combined_exp.ranked_explanation,
                                                  alpha=.2, with_colorbar=False)
            plt.title(f"Explanations for Class {c:02}")
            plt.show()

        # ==== Evaluate Co12 Metrics ====
        # todo: create tqdm bar that tracks each metric progress overall
        metric_kwargs = {"exp": combined_exp, "max_batch_size": batch_size}
        # todo: set leave=False on all tqdm calls within these

        # == Evaluate Correctness ==
        logger.info("Evaluating generated explanations...")
        correctness = Correctness(**metric_kwargs)

        correctness_similarity = correctness.evaluate(
            method="model_randomisation", visualise=visualise,
        )
        # Evaluate Similarity object to array in order given by available_sim_metrics (dictates column order)
        correctness_similarity_vals = evaluate_sim_to_array(correctness_similarity)

        correctness_id_dict = correctness.evaluate(
            method="incremental_deletion",
            deletion_method=deletion_method,
            iterations=deletion_iterations, n_random_rankings=num_random_trials,
            random_seed=random_seed, visualise=visualise,
        )

        # == Evaluate Output Completeness ==
        output_completeness = OutputCompleteness(**metric_kwargs)
        deletion_check = output_completeness.evaluate(
            method="deletion_check", deletion_method=deletion_method,
            threshold=output_completeness_threshold, n_random_rankings=num_random_trials,
            random_seed=random_seed, visualise=visualise,
        )
        preservation_check = output_completeness.evaluate(
            method="preservation_check", deletion_method=deletion_method,
            threshold=output_completeness_threshold, n_random_rankings=num_random_trials,
            random_seed=random_seed, visualise=visualise,
        )

        # == Evaluate Continuity ==
        continuity = Continuity(**metric_kwargs)
        continuity_similarity = continuity.evaluate(
            method="perturbation", visualise=visualise,
            degree=continuity_perturbation_degree, random_seed=random_seed,
        )
        continuity_similarity_vals = evaluate_sim_to_array(continuity_similarity)

        # == Evaluate Contrastivity ==
        contrastivity = Contrastivity(**metric_kwargs)
        contrastivity_similarity = contrastivity.evaluate(
            method="adversarial_attack", visualise=visualise,
        )
        contrastivity_similarity_vals = evaluate_sim_to_array(contrastivity_similarity)

        # == Evaluate Compactness ==
        compactness = Compactness(**metric_kwargs)
        compactness_scores = compactness.evaluate(
            method="threshold", threshold=compactness_threshold, visualise=visualise,
        )

        # ==== Save all results in the dataframe's row for that class ====
        logger.info("Saving calculated evaluation metrics to results dataframe...")
        results_df.loc[dataset.classes[c]] = [
            # Calculate mean similarity across samples; unpack with * array of len(available_sim_metrics) to fill cols
            *correctness_similarity_vals.mean(axis=1),
            # Calculate ratio of AUC for informed deletion / AUC for randomised deletion
            # We expect the AUC of informed deletion < randomised one, so ratio should be small
            (correctness_id_dict["informed"]/correctness_id_dict["random"]).mean(),
            deletion_check.mean(),
            preservation_check.mean(),
            *continuity_similarity_vals.mean(axis=1),
            *contrastivity_similarity_vals.mean(axis=1),
            compactness_scores.mean(),
        ]

    store = pd.HDFStore(str(Path(explainer_name)/"output.h5"))
    store[f"{dataset_name}_{model_name}"] = results_df  # saves results_df object to HDF5 file

else:
    raise RuntimeError("Please run this script from the command line.")
