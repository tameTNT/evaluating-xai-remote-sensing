import captum.attr
import einops
import numpy as np
import shap
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.autonotebook import tqdm


def calculate_shap_values_tensor(
        explain_these: Float[Tensor, "batch_size channels height width"],
        given_this_background: Float[Tensor, "batch_size channels height width"],
        with_this_explainer: captum.attr.DeepLiftShap,
        num_classes: int,
) -> Float[Tensor, "labels batch_size channels height width"]:
    """
    Calculate SHAP values using an explainer for a given model and a given set of images.
    :param explain_these: Tensor of images to explain
    :param given_this_background:
    :param with_this_explainer:
    :param num_classes: num of classes in dataset to explain
    :return:
    """
    model_device = next(with_this_explainer.model.parameters()).device

    shap_vals = torch.zeros(0).to(model_device)
    for i in tqdm(range(num_classes), desc="Calculating SHAP values for class i"):
        vals_for_ith_label = with_this_explainer.attribute(
            explain_these.to(model_device),
            # changing background can use a HUGE amount of memory (50GB for 25 imgs) so limit!
            given_this_background.to(model_device),
            target=torch.tensor(i).to(model_device)
        )

        shap_vals = torch.cat(
            (shap_vals, vals_for_ith_label.unsqueeze(0)),
            dim=0
        ) if shap_vals.size else vals_for_ith_label.unsqueeze(0)

    shap_vals = shap_vals.detach()
    return shap_vals


def prepare_shap_for_image_plot(shap_vals: Float[Tensor, "labels batch_size channels height width"]) -> list:
    return list(einops.rearrange(shap_vals, "l b c h w -> l b h w c").cpu().numpy())


def make_shap_plots(model, shap_vals, for_images, with_labels, split_size, label_classes, device, show_true=True):
    for i, (some_shap_values, some_images, some_labels) in enumerate(zip(
            torch.split(shap_vals, split_size, dim=0),
            torch.split(for_images, split_size, dim=0),
            torch.split(with_labels, split_size, dim=0)
    )):
        print(f"Plotting {i * split_size} to {(i + 1) * split_size - 1}:")
        shap_to_display = list(einops.rearrange(some_shap_values, "b label c h w -> label b h w c").cpu().numpy())
        some_images_to_explain = einops.rearrange(some_images, "b c h w -> b h w c").cpu().numpy()

        predicted = torch.argmax(model(some_images.to(device)), dim=1).cpu().numpy()
        predicted_text_labels = [label_classes[int(i)] for i in predicted]
        true_text_labels = [label_classes[int(i)] for i in some_labels]

        shap.image_plot(shap_to_display, some_images_to_explain, labelpad=0.3, hspace=0.3,
                        labels=np.tile(label_classes, (some_images.shape[0], 1)),
                        true_labels=[f"{pred[:5]}" + (f" ({true[:5]})" if show_true else "") for pred, true in
                                     zip(predicted_text_labels, true_text_labels)],
                        )
