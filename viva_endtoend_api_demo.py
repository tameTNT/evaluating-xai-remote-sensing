import json
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import torch
import numpy as np

import dataset_processing
import helpers
import models
import xai
from evaluate_xai import Similarity
from evaluate_xai.compactness import Compactness
from evaluate_xai.contrastivity import Contrastivity
from evaluate_xai.correctness import Correctness
from evaluate_xai.output_completeness import OutputCompleteness


# PyCharm practicalities
# Ctrl+Shift+./, in increase/decrease font size
# Presentation mode with Ctrl+F5

np.set_printoptions(precision=5, linewidth=150, threshold=150, floatmode="fixed", suppress=True)

# ==== Set script arguments ====
random_seed = 705
dataset_name: dataset_processing.DATASET_NAMES = "UCMerced"
normalisation_type = "scaling"  # only needed for EuroSATRGB/MS

model_name: models.MODEL_NAMES = "ConvNeXtSmall"
batch_size = 12
num_workers = 4

explainer_name: xai.EXPLAINER_NAMES = "GradCAM"
shap_max_evals = 500

deletion_method = "shuffle"
oc_proportion = 0.2
similarity_intersection = 0.1
random_ensemble_size = 5

# will move the model to mps after explanations have been generated
if explainer_name == "PartitionSHAP":  # explanation generation can use mps okay
    use_mps_as_possible = False
    torch_device = helpers.utils.get_torch_device(force_mps=True)
else:
    use_mps_as_possible = True
    torch_device = helpers.utils.get_torch_device()

torch.manual_seed(random_seed)

# ==== Load dataset and corresponding pretrained model ====
# Can be adapted to any model and dataset combination
# (any Model object can be swapped in as can any RSDatasetMixin class)
model_type = models.get_model_type(model_name)
dataset = dataset_processing.get_dataset_object(
    dataset_name, "val", model_type.expected_input_dim,  # use validation set when setting up metrics
    normalisation_type=normalisation_type,
    batch_size=batch_size, num_workers=num_workers, device=torch_device,
    download=False,
)

model_to_explain = model_type(
    pretrained=False, n_input_bands=dataset.N_BANDS, n_output_classes=dataset.N_CLASSES,
)
sim_intersection_m = int(similarity_intersection * (model_to_explain.expected_input_dim ** 2))

# Load pretrained weights
weights_path = json.load(Path("weights_paths.json").open("r"))[dataset_name][model_name]
model_weights_path = helpers.env_var.get_project_root() / "checkpoints" / dataset_name / model_name / weights_path
model_to_explain.load_weights(model_weights_path)

model_to_explain.eval().to(torch_device)


# ==== Select images from the dataset to explain ====
temp_idxs = torch.randint(0, len(dataset), (8,))
imgs_to_explain = torch.stack([dataset[i]["image"] for i in temp_idxs])
print("Image labels:", [dataset.classes[int(dataset[i]["label"])] for i in temp_idxs])
helpers.plotting.show_image(imgs_to_explain)
plt.suptitle("Images to be explained")
plt.show()

# ==== Generate explanation for selected images ====
# An Explainer object takes in a model and is unique for a particular input
explainer = xai.get_explainer_object(
    explainer_name, model=model_to_explain, extra_path=Path(dataset_name + "_demo"),
    attempt_load=imgs_to_explain, batch_size=batch_size,
)

explain_args = {}
if explainer_name == "PartitionSHAP":
    explain_args["shap_batch_size"] = batch_size
    explain_args["max_evals"] = shap_max_evals

if not explainer.has_explanation_for(imgs_to_explain):
    explainer.explain(imgs_to_explain, **explain_args)

# Visualise the generated explanations
helpers.plotting.visualise_importance(imgs_to_explain, explainer.explanation,
                                      alpha=.7, with_colorbar=True, band_idxs=dataset.rgb_indices)
plt.title("Explanations")
plt.show()

helpers.plotting.visualise_importance(imgs_to_explain, explainer.ranked_explanation,
                                      alpha=.7, with_colorbar=True, band_idxs=dataset.rgb_indices)
plt.title("Ranked Explanations")
plt.show()

# ==== Metric evaluation ====
# == Correctness ==
correctness_metric = Correctness(explainer, batch_size=batch_size)

# Model Randomisation
random_model_sim: Similarity = correctness_metric.evaluate(method="model_randomisation", visualise=True)
corr_sim_metrics = random_model_sim(l2_normalise=True, intersection_m=sim_intersection_m, show_scatter=False)
print("\nCorrectness evaluation via model randomisation (↓)")
pprint(corr_sim_metrics)

# Move model to mps after explanations (including for Correctness) have been generated
if use_mps_as_possible:
    explainer.model = explainer.model.to(helpers.utils.get_torch_device(force_mps=True))

# Incremental Deletion (AUC_ratio)
nn_aucs = correctness_metric.evaluate(
    method="incremental_deletion",
    deletion_method=deletion_method,
    iterations=15, n_random_rankings=random_ensemble_size,
    random_seed=42, visualise=True,
)
ratio = nn_aucs["informed"]/nn_aucs["random"]
print("\nCorrectness evaluation via incremental deletion (↓)")
print(ratio)

# == Output Completeness ==
output_completeness_metric = OutputCompleteness(explainer, batch_size=batch_size)

# Deletion Check
deletion_scores = output_completeness_metric.evaluate(
    method="deletion_check", deletion_method=deletion_method, proportion=oc_proportion,
    n_random_rankings=random_ensemble_size, random_seed=42, visualise=True,
)
print("\nOutput completeness evaluation via deletion check (↑)")
print(deletion_scores)

# Preservation Check
preservation_scores = output_completeness_metric.evaluate(
    method="preservation_check", deletion_method=deletion_method, proportion=oc_proportion,
    n_random_rankings=random_ensemble_size, random_seed=42, visualise=True, store_full_data=False
)
print("\nOutput completeness evaluation via preservation check (↑)")
print(preservation_scores)

# print(f"== Extra ==\nOriginal predictions (confidence): "
#       f"{output_completeness_metric.full_data['original_predictions']}")
# #     f" ({output_completeness_metric.full_data['original_pred_confidence']})")
# print(f"Confidence after informed: {output_completeness_metric.full_data['informed_confidences']}")
# print(f"Confidence after random: {output_completeness_metric.full_data['random_confidences']}")

# Contrastivity again needs the model on cuda or cpu
if use_mps_as_possible:
    explainer.model = explainer.model.to(torch_device)

# == Contrastivity ==
contrastivity_metric = Contrastivity(explainer, batch_size=batch_size)
target_sensitivity_sim: Similarity = contrastivity_metric.evaluate(
    method="target_sensitivity", visualise=True,
)
contrastivity_sim_metrics = target_sensitivity_sim(
    l2_normalise=True, intersection_m=sim_intersection_m, show_scatter=False
)
print("\nContrastivity evaluation via adversarial attack (↓)")
pprint(contrastivity_sim_metrics)
print("NB: skipped indices (no adversarial example) are", target_sensitivity_sim.hidden_idxs)

# == Compactness ==
compactness_metric = Compactness(explainer, batch_size=batch_size)
compactness_scores = compactness_metric.evaluate(
    method="threshold", threshold=0.5, visualise=True,
)
print("\nCompactness evaluation via threshold (↑)")
print(compactness_scores)
