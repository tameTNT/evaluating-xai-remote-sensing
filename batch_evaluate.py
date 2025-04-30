"""
This is the primary evaluation script used to generate all explanations and
evaluate them via metrics for the Co-12 properties. This can take some time
for larger datasets and larger models. It should be run from the command line
directly via e.g. `python batch_evaluate.py --help`
"""

import argparse
import platform
import typing as t
from pathlib import Path
import json
import functools
import warnings

import torch
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

torch_device = helpers.utils.get_torch_device(force_mps=True)
logger = helpers.log.main_logger


def get_data_and_model() -> tuple:
    logger.debug("Loading dataset and model...")

    model_type = models.get_model_type(model_name)
    ds = dataset_processing.get_dataset_object(
        dataset_name, "test", model_type.expected_input_dim,
        normalisation_type=normalisation_type, use_resize=True,
        batch_size=batch_size, num_workers=num_workers, device=torch_device,
        download=False,
    )

    weights_path = json.load(Path("weights_paths.json").open("r"))[dataset_name][model_name]
    model_weights_path = checkpoints_path / dataset_name / model_name / weights_path

    model = model_type(
        pretrained=False, n_input_bands=ds.N_BANDS, n_output_classes=ds.N_CLASSES,
    )
    model.load_weights(model_weights_path)
    model.eval().to(torch_device)

    return ds, model


def get_hdf5():
    if not h5_output_path.exists():
        h5_output_path.mkdir(parents=True)

    # futurefix: there is a slim chance that the file is already open in another process - handle this gracefully
    store_for_explainer_name = pd.HDFStore(f"{h5_output_path / h5_output_name}.h5", mode="a")
    logger.debug(f"Opened HDF5 file {h5_output_path / h5_output_name}.h5.")
    return store_for_explainer_name


def get_explainer_args() -> dict:
    explain_args = {}
    if explainer_name == "PartitionSHAP":
        explain_args["shap_batch_size"] = shap_batch_size
        explain_args["max_evals"] = shap_max_evals
        explain_args["num_mp_processes"] = shap_multi_processes

    return explain_args


def generate_explanations(_for_idxs: np.array, class_idx: int) -> list[xai.Explainer]:
    num_batches = int(np.ceil(len(_for_idxs) / batch_size))
    explainers = []
    for i in tqdm(range(num_batches), desc="Generating explanations", mininterval=2,
                  unit="batch", ncols=110, leave=False):
        idxs_for_batch = _for_idxs[i*batch_size:(i+1)*batch_size]
        batch = torch.stack([dataset[j]["image"] for j in idxs_for_batch])

        explainer = xai.get_explainer_object(
            explainer_name, model_to_explain,
            extra_path=Path(dataset_name) / f"c{class_idx:02}" / f"b{i:03}",
            attempt_load=batch, batch_size=batch_size,
        )
        if not explainer.has_explanation_for(batch):
            logger.debug(f"No existing explanation for batch {i} of class {class_idx}. Generating new ones.")
            explainer.explain(batch, **get_explainer_args())
        explainers.append(explainer)

    return explainers


def build_parameters_dict() -> dict:
    return {
        "shap_max_evals": shap_max_evals,
        "output_completeness_proportion": output_completeness_proportion,
        "continuity_perturbation_degree": continuity_perturbation_degree,
        "compactness_threshold": compactness_threshold,
        "samples_per_class": samples_per_class,
        "deletion_method": deletion_method,
        "deletion_iterations": deletion_iterations,
        "num_random_trials": num_random_trials,
        "similarity_intersection_proportion": similarity_intersection_proportion,
        "min_samples_for_similarity": min_samples_for_similarity,
    }


def evaluate_sim_to_array(sim: Similarity) -> np.array:
    sim_return_dict = sim(l2_normalise=True, intersection_m=similarity_intersection_m)
    return np.stack([sim_return_dict[m] for m in available_sim_metrics])


if __name__ == "__main__":
    # ==== Parse command line arguments ====
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
    script_meta_group.add_argument(
        "--h5_output_name",
        type=str,
        default="evaluation_output",
        help="Name of the HDF5 file to store the evaluation results in. Defaults to 'evaluation_output'.",
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
    options_group.add_argument(
        "--shap_max_evals",
        type=int,
        default=10000,
        help="Maximum number of evaluations to use for SHAP methods. "
             "Defaults to 10000.",
    )
    options_group.add_argument(
        "--shap_batch_size",
        type=int,
        default=0,
        help="Batch size to use for SHAP methods. Note that no gradients are stored so this "
             "can be larger than the regular batch size. If 0, defaults to the same as the regular batch size.",
    )
    options_group.add_argument(
        "--shap_multi_processes",
        type=int,
        default=1,
        help="⚠️This option is still under development. Number of multi-processes to use to speed up SHAP methods. "
             "Defaults to 1 (no multiprocessing).",
    )
    metric_options = parser.add_argument_group("Metric Options",
                                               "Options for specific evaluation metrics.")
    metric_options.add_argument(
        "--output_completeness_proportion",
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
        choices=["blur", "inpaint", "nn", "shuffle"],
        help="Method to use for deletion/perturbation-related evaluation methods. "
             "Defaults to 'shuffle'.",
    )
    shared_options.add_argument(
        "--deletion_iterations",
        type=int,
        default=15,
        help="Number of iterations to use for the incremental methods. "
             "This increases memory footprint significantly. Defaults to 15.",
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
        help="Proportion of image pixels to use for the intersection similarity metric. "
             "Defaults to 0.1.",
    )
    shared_options.add_argument(
        "--min_samples_for_similarity",
        type=int,
        default=10,
        help="Minimum number of samples to allow to calculate a similarity metric. "
             "Defaults to 5.",
    )

    args: argparse.Namespace = parser.parse_args()
    print("Got args:", args, "\n")

    random_seed: int = args.random_seed
    checkpoints_path: Path = args.checkpoints_path.expanduser().resolve()
    num_workers: int = args.num_workers
    batch_size: int = args.batch_size
    visualise: bool = args.visualise
    h5_output_name: str = args.h5_output_name

    dataset_name: dataset_processing.DATASET_NAMES = args.dataset_name
    normalisation_type: str = args.normalisation_type
    model_name: models.MODEL_NAMES = args.model_name
    explainer_name: xai.EXPLAINER_NAMES = args.explainer_name
    shap_max_evals: int = args.shap_max_evals
    shap_batch_size: int = args.shap_batch_size
    shap_batch_size = batch_size if shap_batch_size == 0 else shap_batch_size
    shap_multi_processes: int = args.shap_multi_processes

    output_completeness_proportion: float = args.output_completeness_proportion
    continuity_perturbation_degree: float = args.continuity_perturbation_degree
    compactness_threshold: float = args.compactness_threshold

    samples_per_class: int = args.samples_per_class
    deletion_method: evaluate_xai.deletion.METHODS = args.deletion_method
    deletion_iterations: int = args.deletion_iterations
    num_random_trials: int = args.num_random_trials
    similarity_intersection_proportion: float = args.similarity_intersection_proportion
    min_samples_for_similarity: int = args.min_samples_for_similarity

    # ==== Begin actual script execution using parsed args ====
    print(f"Logging to {logger.handlers[0].baseFilename}. See file for details.\n")
    logger.info(f"Successfully got script arguments: {args}.")

    torch.manual_seed(random_seed)  # set torch seed here before getting the dataset to ensure any splits are the same

    dataset, model_to_explain = get_data_and_model()

    similarity_intersection_m = int(similarity_intersection_proportion * model_to_explain.expected_input_dim ** 2)
    available_sim_metrics = t.get_args(evaluate_xai.SIMILARITY_METRICS)

    # == Build the output results DataFrame and load any existing data if it exists ==
    results_df = pd.DataFrame(columns=[
        # ↓, low similarity
        *[f"correctness : randomised_model_similarity : {metric_name}" for metric_name in available_sim_metrics],
        # ↓, best is 0
        "correctness : incremental_deletion_auc_ratio",
        # ↑, best is 1
        "output_completeness : deletion_check_conf_drop",
        # ↑, best is 1
        "output_completeness : preservation_check_conf_drop",
        # ↑, high similarity
        *[f"continuity : perturbation_similarity : {metric_name}" for metric_name in available_sim_metrics],
        # ↓, low similarity
        *[f"contrastivity : adversarial_attack_similarity : {metric_name}" for metric_name in available_sim_metrics],
        # ↑, near 1
        "compactness : threshold_score"
    ], index=dataset.classes, dtype=float)  # new row for each dataset class

    h5_output_path = helpers.env_var.get_project_root() / "results" / explainer_name

    ds_model_df_name = f"{dataset_name}_{model_name}"
    json_parameters_path = h5_output_path / f"{h5_output_name}_parameters.json"
    h5_store = get_hdf5()
    df_already_exists = ds_model_df_name in h5_store
    if not df_already_exists:
        h5_store[ds_model_df_name] = results_df
        h5_store.close()  # close the HDF5 file again asap after reading/writing to it!
        json.dump(build_parameters_dict(), json_parameters_path.open("w+"), indent=4)
    else:
        assert np.array_equal(h5_store[ds_model_df_name].columns, results_df.columns), \
            f"Unexpected columns in existing dataframe."
        h5_store.close()
        stored_parameters = json.load(json_parameters_path.open("r"))
        current_parameters = build_parameters_dict()
        if stored_parameters != current_parameters:
            for key, value in current_parameters.items():
                if key not in stored_parameters or stored_parameters[key] != value:
                    warning_txt = (f"Parameter '{key}' has changed from "
                                   f"'{stored_parameters[key] if key in stored_parameters else '[not present]'}' "
                                   f"to '{value}'.")
                    warnings.warn(warning_txt)
                    logger.warning(warning_txt)
            logger.warning("Some parameters have changed since the last evaluation. Continuing regardless.")

    # ==== Iterate over every class in the dataset ====
    classes = np.array([class_ for _, class_ in dataset.imgs])
    for c in tqdm(range(dataset.N_CLASSES), ncols=110, desc="xAI per class"):
        current_class_name = dataset.classes[c]
        h5_store = get_hdf5()
        existing_df_row = h5_store[ds_model_df_name].loc[current_class_name]
        h5_store.close()
        if existing_df_row.isna().sum() == 0:  # no NaN values in row
            logger.info(f"All metrics already calculated and saved for class {c:02}. Skipping.")
            continue

        # Use the same random seed for each class each time (enables repeatable resuming runs)
        torch.manual_seed(random_seed)
        np_rng = np.random.default_rng(random_seed)

        class_idxs = np.where(classes == c)[0]

        num_samples_to_take = samples_per_class
        if len(class_idxs) < num_samples_to_take:
            logger.warning(f"samples_per_class={samples_per_class} > num samples for class {c} ({len(class_idxs)}). "
                           f"Using {len(class_idxs)}.")
            num_samples_to_take = len(class_idxs)
        elif samples_per_class <= 0:
            num_samples_to_take = len(class_idxs)
        class_idxs_sampled = np_rng.choice(class_idxs, num_samples_to_take, replace=False)

        explainers_for_c = generate_explanations(class_idxs_sampled, c)
        # Combine all the explainers for this class into one
        combined_exp = functools.reduce(lambda x, y: x | y, explainers_for_c)
        # update extra_path to ensure each combined explainer is saved on an overall per-class basis (not per-batch)
        combined_exp.extra_path = Path(dataset_name) / f"c{c:02}" / "combined"

        if visualise:
            helpers.plotting.visualise_importance(combined_exp.input,
                                                  combined_exp.ranked_explanation,
                                                  alpha=.2, with_colorbar=False, band_idxs=dataset.rgb_indices)
            plt.title(f"Explanations for Class {c:02}")
            plt.show()

        # ==== Evaluate Metrics for Co12 Properties ====
        with tqdm(total=7, ncols=110, desc="Calculating metrics", leave=False) as metric_pbar:
            metric_kwargs = {"exp": combined_exp, "batch_size": batch_size}
            # Array of -inf values to use for voided similarity metrics
            sim_inf_array = np.zeros((len(available_sim_metrics), combined_exp.input.shape[0])) - np.inf

            # == Evaluate Correctness ==
            logger.info("Evaluating generated explanations...")
            correctness = Correctness(**metric_kwargs)

            correctness_similarity = correctness.evaluate(
                method="model_randomisation", visualise=visualise,
            )
            # Evaluate the Similarity object into an array
            # in the order given by available_sim_metrics (dictates column order)
            correctness_similarity_vals = evaluate_sim_to_array(correctness_similarity)
            metric_pbar.update()

            correctness_id_dict = correctness.evaluate(
                method="incremental_deletion",
                deletion_method=deletion_method,
                iterations=deletion_iterations, n_random_rankings=num_random_trials,
                random_seed=random_seed, visualise=visualise,
            )
            metric_pbar.update()

            # == Evaluate Output Completeness ==
            output_completeness = OutputCompleteness(**metric_kwargs)
            deletion_check = output_completeness.evaluate(
                method="deletion_check", deletion_method=deletion_method,
                proportion=output_completeness_proportion, n_random_rankings=num_random_trials,
                random_seed=random_seed, visualise=visualise,
            )
            metric_pbar.update()

            preservation_check = output_completeness.evaluate(
                method="preservation_check", deletion_method=deletion_method,
                proportion=output_completeness_proportion, n_random_rankings=num_random_trials,
                random_seed=random_seed, visualise=visualise,
            )
            metric_pbar.update()

            # == Evaluate Continuity ==
            continuity = Continuity(**metric_kwargs)
            continuity_similarity = continuity.evaluate(
                method="perturbation", visualise=visualise,
                degree=continuity_perturbation_degree, random_seed=random_seed,
            )
            if len(continuity_similarity.return_idxs) < min_samples_for_similarity:
                logger.warning(f"Model outputs changed on too many perturbed samples for class {c} meaning there are "
                               f"not enough samples for min_samples threshold ({min_samples_for_similarity}). "
                               "Skipping continuity evaluation for this class: using -inf values.")
                continuity_similarity_vals = sim_inf_array
            else:
                continuity_similarity_vals = evaluate_sim_to_array(continuity_similarity)
            metric_pbar.update()

            # == Evaluate Contrastivity ==
            contrastivity = Contrastivity(**metric_kwargs)
            contrastivity_similarity = contrastivity.evaluate(
                method="target_sensitivity", visualise=visualise,
            )
            if len(contrastivity_similarity.return_idxs) < min_samples_for_similarity:
                logger.warning(f"Not a sufficient number of successful adversarial attacks for class {c} to "
                               f"meet min_samples threshold ({min_samples_for_similarity}). "
                               "Skipping contrastivity evaluation for this class: using -inf values.")
                contrastivity_similarity_vals = sim_inf_array
            else:
                contrastivity_similarity_vals = evaluate_sim_to_array(contrastivity_similarity)
            metric_pbar.update()

            # == Evaluate Compactness ==
            compactness = Compactness(**metric_kwargs)
            compactness_scores = compactness.evaluate(
                method="threshold", threshold=compactness_threshold, visualise=visualise,
            )
            metric_pbar.update()

        # ==== Save all results in the dataframe's row for that class ====
        logger.info("Saving calculated evaluation metrics to results dataframe...")
        h5_store = get_hdf5()
        # Use pd.to_numeric (converts numpy array objects) since we require numeric
        # values for HDF5 for fast saving/loading
        # Directly update target row only (preserve other rows)
        updated_df = h5_store[ds_model_df_name]
        updated_df.loc[current_class_name] = [
            # Calculate mean similarity across samples;
            # unpack with * the array of len(available_sim_metrics) to fill cols
            *correctness_similarity_vals.mean(axis=1),
            # Calculate the ratio of AUC for informed deletion / AUC for randomised deletion
            # We expect the AUC of informed deletion < randomised one, so the ratio should be small
            (correctness_id_dict["informed"]/correctness_id_dict["random"]).mean(),
            deletion_check.mean(),
            preservation_check.mean(),
            *continuity_similarity_vals.mean(axis=1),
            *contrastivity_similarity_vals.mean(axis=1),
            compactness_scores.mean(),
        ]
        h5_store[ds_model_df_name] = updated_df
        h5_store.close()

    logger.info(f"Script execution complete.")
