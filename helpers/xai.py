import einops
import numpy as np
import shap
import torch
from tqdm.autonotebook import tqdm


def build_shap_vals_tensor(explain_these, given_this_background, with_this_explainer, num_classes, device):
    vals = torch.zeros(0).to(device)
    for to_be_explained in tqdm(explain_these, leave=True):
        image_vals = torch.zeros(0).to(device)
        for i in tqdm(range(num_classes), leave=True):
            vals_for_class_i = with_this_explainer.attribute(to_be_explained.unsqueeze(0).to(device),
                                                             given_this_background,
                                                             target=torch.tensor(i).to(device))
            image_vals = torch.cat((image_vals, vals_for_class_i), dim=0) if image_vals.size else vals_for_class_i
        vals = torch.cat((vals, image_vals.unsqueeze(0)), dim=0)

    vals = vals.detach()
    return vals


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
