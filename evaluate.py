import json
from pathlib import Path

# from remote_plot import plt
import matplotlib.pyplot as plt
import safetensors.torch as st
import torch

import dataset_processing
import helpers
import models
from evaluate_xai.correctness import Correctness
from xai.shap_method import SHAPExplainer

# plt.port = 36422

random_seed = 42
dataset_name = "EuroSATRGB"
normalisation_type = "scaling"
use_resize = True
batch_size = 32

model_name = "ResNet50"
num_workers = 4

logger = helpers.log.get_logger("main")

torch_device = helpers.utils.get_torch_device()
torch.manual_seed(random_seed)

model_type = models.get_model_type(model_name)
dataset = dataset_processing.get_dataset_object(
    dataset_name, "val", model_type.expected_input_dim,  # todo: switch to test
    normalisation_type=normalisation_type, use_resize=use_resize,
    batch_size=batch_size, num_workers=num_workers, device=torch_device,
)

logger.info("Creating model and loading pretrained weights.")

weights_paths = json.load(Path("weights_paths.json").open("r"))[dataset_name][model_name]
model_weights_path = (helpers.env_var.get_project_root() / weights_paths)

model_to_explain = model_type(
    pretrained=False, n_input_bands=dataset.N_BANDS, n_output_classes=dataset.N_CLASSES,
)
st.load_model(model_to_explain, model_weights_path)
model_to_explain.eval().to(torch_device)
logger.info(f"Loaded weights from {model_weights_path} successfully.")

temp_idxs = [481, 4179, 3534, 2369, 2338, 4636,  464, 3765, 1087,  508]
# random_idxs = torch.randint(0, len(dataset), (10,))
imgs_to_explain = torch.stack([dataset[i]["image"] for i in temp_idxs])
helpers.plotting.show_image(imgs_to_explain)
plt.show()

# todo: support saving/loading large batches of explanations
#  rather than needing new obj each time for each batch
shap_explainer = SHAPExplainer(model_to_explain, attempt_load=imgs_to_explain)
if not shap_explainer.has_explanation_for(imgs_to_explain):
    logger.info(f"No existing explanation for imgs_to_explain. Generating a new one.")
    shap_explainer.explain(imgs_to_explain)
else:
    logger.info(f"Existing explanation found for imgs_to_explain.")

helpers.plotting.visualise_importance(imgs_to_explain, shap_explainer.ranked_explanation,
                                      alpha=.2, with_colorbar=False)
plt.show()

correctness_metric = Correctness(shap_explainer, max_batch_size=batch_size)
similarity = correctness_metric.evaluate(method="model_randomisation")
metrics = similarity(l2_normalise=True, intersection_k=5000)
print("Correctness evaluation via model randomisation", metrics)
