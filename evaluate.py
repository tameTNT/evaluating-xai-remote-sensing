from pathlib import Path

import safetensors.torch as st
import torch

import dataset_processing
from evaluate_xai.correctness import Correctness
from helpers.models import FineTunedResNet50
from xai.shap_method import SHAPExplainer

model_weights_path = Path('/home2/jgcw74/l3_project/checkpoints/resnet50/FineTunedResNet50_final_ft_weights(0.989).st')
# todo: modularise get_model_type functions from train.py
model_to_explain = FineTunedResNet50(
    pretrained=False, n_input_bands=3, n_output_classes=10
)
st.load_model(model_to_explain, model_weights_path)
model_to_explain.eval()

random_seed = 42
torch.manual_seed(random_seed)

dataset = dataset_processing.eurosat.EuroSATRGB(
    split="test", image_size=model_to_explain.expected_input_dim
)
imgs_to_explain = torch.stack(
    [dataset[i]["image"] for i in torch.randint(0, len(dataset), (5,))]
)

# todo: support saving/loading large batches of explanations
#  rather than needing new obj each time for each batch
shap_explainer = SHAPExplainer(model_to_explain)
shap_explainer.explain(imgs_to_explain)

correctness_metric = Correctness(shap_explainer)
similarity = correctness_metric.evaluate()
l2_args = {"normalise": True}
metrics = similarity(l2_args=l2_args)
print("L2 distance: ", metrics["l2_distance"])
