import json
from pathlib import Path

import safetensors.torch as st
import torch

import dataset_processing
import helpers
import models
from evaluate_xai.correctness import Correctness
from xai.shap_method import SHAPExplainer

random_seed = 42
dataset_name = "EuroSATRGB"
normalisation_type = "scaling"
use_resize = False
batch_size = 32

model_name = "ResNet50"
num_workers = 4

logger = helpers.log.get_logger("main")

torch_device = helpers.utils.get_torch_device()
torch.manual_seed(random_seed)

model_type = models.get_model_type(model_name)
dataset = dataset_processing.get_dataset_object(
    dataset_name, "test", model_type.expected_input_dim,
    normalisation_type=normalisation_type, use_resize=use_resize,
    batch_size=batch_size, num_workers=num_workers, device=torch_device,
)

logger.info("Creating model and loading pretrained weights.")

weights_paths = json.load(Path("weights_paths.json").open("r"))
model_weights_path = Path(weights_paths[dataset_name][model_name]).expanduser()

model_to_explain = model_type(
    pretrained=False, n_input_bands=dataset.N_BANDS, n_output_classes=dataset.N_CLASSES,
)
st.load_model(model_to_explain, model_weights_path)
model_to_explain.eval().to(torch_device)
logger.info(f"Loaded weights from {model_weights_path} successfully.")


imgs_to_explain = torch.stack(
    [dataset[i]["image"] for i in torch.randint(0, len(dataset), (5,))]
)

# todo: support saving/loading large batches of explanations
#  rather than needing new obj each time for each batch
shap_explainer = SHAPExplainer(model_to_explain, attempt_load=True)
shap_explainer.explain(imgs_to_explain)

correctness_metric = Correctness(shap_explainer)
similarity = correctness_metric.evaluate()
l2_args = {"normalise": True}
metrics = similarity(l2_args=l2_args)
print("L2 distance: ", metrics["l2_distance"])
