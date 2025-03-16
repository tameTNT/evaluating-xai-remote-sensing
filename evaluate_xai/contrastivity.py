import typing as t

import foolbox
import matplotlib.pyplot as plt
import torch
import einops
import numpy as np

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
            img_bounds: tuple[float, float] = (-1, 1),
            attack: foolbox.attacks.Attack = foolbox.attacks.LinfDeepFoolAttack,
            attack_kwargs: dict = None,
            **kwargs,
    ) -> Similarity:

        if attack_kwargs is None:
            attack_kwargs = {}

        original_preds = self.run_model(self.exp.input).argmax(1)
        # We don't care about the images' true labels,
        # just the model's original predictions (and associated explanation)
        criteria = foolbox.criteria.Misclassification(torch.from_numpy(original_preds).to(self.exp.device))

        # check for existing generated adversarial images to save having to regenerate
        need_to_generate = True
        previous_adv_output_path = self.exp.save_path / "adversarial_examples.npz"
        if previous_adv_output_path.exists():
            logger.debug(f"Adversarial images potentially already generated, loading from {previous_adv_output_path}.")
            with np.load(previous_adv_output_path) as data:
                clipped_adv_imgs = torch.from_numpy(data["clipped_adv_imgs"]).to(self.exp.device)
                temp = torch.from_numpy(data["original_imgs"]).to(self.exp.device)

            if torch.equal(temp, self.exp.input):
                logger.info("Saved adversarial images match current explanation input. Loaded.")
                need_to_generate = False
            else:
                logger.warning("Saved adversarial images do not match current explanation input.")

        # if no existing adversarial images, generate them
        if need_to_generate:
            logger.info("No existing adversarial images. Generating new ones via foolbox.")
            foolbox_model = foolbox.PyTorchModel(self.exp.model, bounds=img_bounds)
            # noinspection PyCallingNonCallable
            _, clipped_adv_imgs, success = attack(**attack_kwargs)(
                foolbox_model, self.exp.input, criteria, epsilons=0.01,
            )

            np.savez_compressed(previous_adv_output_path,
                                clipped_adv_imgs=clipped_adv_imgs.numpy(force=True),
                                original_imgs=self.exp.input.numpy(force=True))
            logger.debug(f"Saved generated adversarial images to {previous_adv_output_path}.")

        # noinspection PyUnboundLocalVariable
        adv_preds = self.run_model(clipped_adv_imgs).argmax(1)
        logger.debug(f"{len(clipped_adv_imgs)} adversarial examples (og->adv): "
                     f"{original_preds}->{adv_preds}.")

        # Only compare similarity of explanations where the model predictions changed
        # since the explanations should reflect the underlying model (and change a lot)
        # noinspection PyTypeChecker
        success: np.ndarray[bool] = original_preds != adv_preds
        if not success.all():
            logger.warning(f"Failed to successfully load/generate truly adversarial images for "
                           f"idxs={np.flatnonzero(~success)}.")

        if self.visualise:
            stacked_samples = einops.rearrange(
                torch.stack([self.exp.input, clipped_adv_imgs]), "i n c h w -> n c (i h) w")
            helpers.plotting.show_image(stacked_samples)
            plt.title("Original/Adversarial images")
            plt.show()

        exp_for_adv = self.get_sub_explainer("adversarial", clipped_adv_imgs)

        if self.visualise:
            self.compare_sub_explainer(
                exp_for_adv,
                title="Explanation on original/adversarial input"
            )

        return Similarity(self.exp, exp_for_adv, mask=success)
