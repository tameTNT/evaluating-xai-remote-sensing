import typing as t

import foolbox
import matplotlib.pyplot as plt
import numpy as np
import torch
import einops
from jaxtyping import Float

import helpers
from . import Co12Metric, Similarity

logger = helpers.log.get_logger("main")


class Contrastivity(Co12Metric):
    # todo: add docstrings based on paper

    def evaluate(
            self,
            method: t.Literal["adversarial_attack"],
            **kwargs,
    ) -> Similarity:
        super().evaluate(method, **kwargs)

        if method == "adversarial_attack":
            return self._adversarial_attack(**kwargs)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

    def _adversarial_attack(
            self,
            true_labels: Float[torch.Tensor, "n_samples"],
            img_bounds: tuple[float, float] = (-1, 1),
            attack: foolbox.attacks.Attack = foolbox.attacks.LinfDeepFoolAttack,
            attack_kwargs: dict = None,
            **kwargs,
    ) -> Similarity:
        if attack_kwargs is None:
            attack_kwargs = {}

        foolbox_model = foolbox.PyTorchModel(self.exp.model, bounds=img_bounds)
        criteria = foolbox.criteria.Misclassification(true_labels.to(self.exp.device))
        # noinspection PyCallingNonCallable
        _, clipped_adv_imgs, success = attack(**attack_kwargs)(
            foolbox_model, self.exp.input, criteria, epsilons=0.01,
        )
        if not success.all():
            logger.warning(f"Failed to successfully generate adversarial images for "
                           f"idxs={(~success).argwhere().flatten().numpy(force=True)}.")

        if self.visualise:
            stacked_samples = einops.rearrange(
                torch.stack([self.exp.input, clipped_adv_imgs]), "i n c h w -> n c (i h) w")
            helpers.plotting.show_image(stacked_samples)
            plt.title("Original/Adversarial images")
            plt.show()

        adv_preds = self.run_model(clipped_adv_imgs).argmax(1)
        logger.debug(f"Generated {len(clipped_adv_imgs)} adversarial examples (og->adv): "
                     f"{criteria.labels.numpy()}->{adv_preds}.")

        exp_for_adv = self.generate_sub_explainer("adversarial", clipped_adv_imgs)

        return Similarity(self.exp, exp_for_adv)
