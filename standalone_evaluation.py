import json
from pathlib import Path

# from remote_plot import plt
import matplotlib.pyplot as plt
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image

import dataset_processing
import helpers
import models
import xai
from evaluate_xai import Similarity
from evaluate_xai.compactness import Compactness
from evaluate_xai.continuity import Continuity
from evaluate_xai.contrastivity import Contrastivity
from evaluate_xai.correctness import Correctness
from evaluate_xai.output_completeness import OutputCompleteness

# plt.port = 36422

# ==== Set up script arguments ====
random_seed = 42
dataset_name: dataset_processing.DATASET_NAMES = "EuroSATRGB"
normalisation_type = "scaling"
use_resize = True
batch_size = 32

model_name: models.MODEL_NAMES = "ConvNeXtSmall"
num_workers = 4

explainer_name: xai.EXPLAINER_NAMES = "KPCACAM"
shap_max_evals = 500

logger = helpers.log.main_logger

torch_device = helpers.utils.get_torch_device(force_mps=True)
torch.manual_seed(random_seed)

# ==== Load dataset and corresponding pretrained model ====
model_type = models.get_model_type(model_name)
dataset = dataset_processing.get_dataset_object(
    dataset_name, "val", model_type.expected_input_dim,  # use validation set when setting up metrics
    normalisation_type=normalisation_type, use_resize=use_resize,
    batch_size=batch_size, num_workers=num_workers, device=torch_device,
    download=False,
)

logger.info("Creating model and loading pretrained weights.")

weights_path = json.load(Path("weights_paths.json").open("r"))[dataset_name][model_name]
model_weights_path = helpers.env_var.get_project_root() / "checkpoints" / dataset_name / model_name / weights_path

model_to_explain = model_type(
    pretrained=False, n_input_bands=dataset.N_BANDS, n_output_classes=dataset.N_CLASSES,
)
model_to_explain.load_weights(model_weights_path)
model_to_explain.eval().to(torch_device)

# ==== Select images from dataset to explain ====
# temp_idxs = [481, 4179, 3534, 2369, 2338, 4636,  464, 3765, 1087,  508]
temp_idxs = [4179, 3534, 2338]
# temp_idxs = torch.randint(0, len(dataset), (10,))
imgs_to_explain = torch.stack([dataset[i]["image"] for i in temp_idxs])
helpers.plotting.show_image(imgs_to_explain, final_fig_size=(8., 4.))  # , normalisation_type="channel")
plt.suptitle("Images to be explained")
plt.show()

# ==== Generate explanation for selected images ====
explainer = xai.get_explainer_object(
    explainer_name, model=model_to_explain, extra_path=Path(dataset_name),
    # attempt_load=imgs_to_explain,
    # batch_size=batch_size,
)

explain_args = {}
if explainer_name == "PartitionSHAP":
    explain_args["shap_batch_size"] = batch_size
    explain_args["max_evals"] = shap_max_evals

if not explainer.has_explanation_for(imgs_to_explain):
    logger.info(f"No existing explanation for imgs_to_explain. Generating a new one.")
    explainer.explain(imgs_to_explain, **explain_args)
else:
    logger.info(f"Existing explanation found for imgs_to_explain.")

plt.imshow(show_cam_on_image(
    (imgs_to_explain[0, [2, 1, 0]].numpy()*2 + 1).clip(0, 1).transpose(1, 2, 0),
    explainer.explanation[0], use_rgb=True)
)
plt.show()
# move channel to final dimension
helpers.plotting.visualise_importance(imgs_to_explain, explainer.explanation,
                                      alpha=.2, with_colorbar=True, band_idxs=dataset.rgb_indices)
plt.suptitle("Explanations")
plt.show()

helpers.plotting.visualise_importance(imgs_to_explain, explainer.ranked_explanation,
                                      alpha=.2, with_colorbar=True, band_idxs=dataset.rgb_indices)
plt.suptitle("Ranked explanations being evaluated")
plt.show()

# ==== Evaluate explanation using Co12 Metrics ====
deletion_method = "blur"  # "shuffle" or "nn" works best here in most cases
# Applying deletion method to sat img with large 'class regions' is hard

# == Correctness ==
correctness_metric = Correctness(explainer, batch_size=batch_size)

# Model Randomisation
corr_similarity: Similarity = correctness_metric.evaluate(method="model_randomisation", visualise=True)
corr_sim_metrics = corr_similarity(l2_normalise=True, intersection_m=5000, show_scatter=False)
print("Correctness evaluation via model randomisation (↓)", corr_sim_metrics)

# Incremental Deletion
nn_aucs = correctness_metric.evaluate(
    method="incremental_deletion",
    deletion_method=deletion_method,
    iterations=10, n_random_rankings=3,
    random_seed=42, visualise=True,
)
print("Correctness evaluation via incremental deletion (↓)", nn_aucs["informed"]/nn_aucs["random"])

# == Output Completeness ==
output_completeness_metric = OutputCompleteness(explainer, batch_size=batch_size)
proportion = 0.2

# Deletion Check
drop_in_confidence = output_completeness_metric.evaluate(
    method="deletion_check", deletion_method=deletion_method, proportion=proportion,
    n_random_rankings=5, random_seed=42, visualise=True,
)
print("Output completeness evaluation via deletion check (↑)", end=" ")
print(", ".join([f"{d:.3f}" for d in drop_in_confidence]))

# Preservation Check
drop_in_confidence = output_completeness_metric.evaluate(
    method="preservation_check", deletion_method=deletion_method, proportion=proportion,
    n_random_rankings=5, random_seed=42, visualise=True,
)
print("Output completeness evaluation via preservation check (↑)", end=" ")
print(", ".join([f"{d:.3f}" for d in drop_in_confidence]))

# == Continuity ==
continuity_metric = Continuity(explainer, batch_size=batch_size)

# Model Randomisation
similarity = continuity_metric.evaluate(
    method="perturbation", visualise=True,
    degree=0.15, random_seed=42,
)
print(f"{len(similarity.hidden_idxs)} predictions changed after perturbation at "
      f"indices: {similarity.hidden_idxs}")
cont_sim_metrics = similarity(l2_normalise=True, intersection_m=5000)
print("Continuity evaluation via image perturbation (↑)", cont_sim_metrics)

# == Contrastivity ==
contrastivity_metric = Contrastivity(explainer, batch_size=batch_size)

# Adversarial Attack
similarity = contrastivity_metric.evaluate(
    method="target_sensitivity", visualise=True,
)
contrastivity_sim_metrics = similarity(l2_normalise=True, intersection_m=5000, show_scatter=True)
print("Contrastivity evaluation via adversarial attack (↓)", contrastivity_sim_metrics)

# == Compactness ==
compactness_metric = Compactness(explainer, batch_size=batch_size)
compactness_scores = compactness_metric.evaluate(
    method="threshold", threshold=0.5, visualise=True,
)
print("Compactness evaluation via threshold (↑)", compactness_scores)
