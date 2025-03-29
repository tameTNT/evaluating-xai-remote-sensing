import typing as t

import foolbox
import matplotlib.pyplot as plt
import torch
import einops
import numpy as np

import helpers
from . import Co12Metric, Similarity

logger = helpers.log.main_logger


class Contrastivity(Co12Metric):
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
            # use the default number of steps
            # We're only working with 10-20 class datasets so 5 candidates is reasonable
            attack_kwargs = {"steps": 50, "candidates": 5}

        original_preds = self.run_model(self.exp.input).argmax(1)

        # check for existing generated adversarial images to save having to regenerate
        need_to_generate = True
        # adversarial images are specific to a trained model (using their gradients)
        previous_adv_output_path = (self.exp.save_path /
                                    f"{self.exp.model.__class__.__name__}_adversarial_examples.npz")

        if previous_adv_output_path.exists():
            logger.debug(f"Adversarial images potentially already generated, loading from {previous_adv_output_path}.")
            with np.load(previous_adv_output_path) as data:
                clipped_adv_imgs = torch.from_numpy(data["clipped_adv_imgs"]).to(self.exp.device)
                temp = torch.from_numpy(data["original_imgs"]).to(self.exp.device)

            if temp.shape != self.exp.input.shape:
                num_desired = self.exp.input.shape[0]
            else:
                num_desired = temp.shape[0]

            if torch.equal(temp[:num_desired], self.exp.input):
                logger.info("Saved adversarial images match current explanation input. "
                            "Loaded and skipping generation.")
                clipped_adv_imgs = clipped_adv_imgs[:num_desired]
                logger.warning(f"Only needed to use the first {num_desired} images from the saved adversarial images.")
                need_to_generate = False
            else:
                logger.warning(f"Saved adversarial images (shape={temp.shape}) do not match "
                               f"current explanation input (shape={self.exp.input.shape}).")

        # if no existing adversarial images, generate them
        if need_to_generate:
            img_outputs = []
            attack_batch_size = self.max_batch_size
            if attack_batch_size == 0:
                attack_batch_size = self.exp.input.shape[0]

            logger.info(f"No existing adversarial images. "
                        f"Generating new ones via foolbox with batch size {attack_batch_size}.")
            foolbox_model = foolbox.PyTorchModel(self.exp.model, device=self.exp.device, bounds=img_bounds)

            for batch_input, batch_preds in zip(
                    helpers.utils.make_device_batches(
                        self.exp.input, attack_batch_size, self.exp.device),
                    helpers.utils.make_device_batches(
                        torch.from_numpy(original_preds), attack_batch_size, self.exp.device),
            ):
                # We don't care about the images' true labels,
                # just the model's original predictions (and associated explanation)
                batch_criteria = foolbox.criteria.Misclassification(batch_preds)

                # Images may be out of bounds slightly due to naive normalisation
                batch_min, batch_max = batch_input.min(), batch_input.max()
                if batch_min < img_bounds[0] or batch_max > img_bounds[1]:
                    logger.warning(f"Input images were outside the bounds {img_bounds}: "
                                   f"min={batch_min}, max={batch_max}. Clamping.")
                    batch_input = batch_input.clamp(img_bounds[0], img_bounds[1])

                # WARNING: there is a rogue call to the base logging.info() on
                # line 127 of foolbox/attacks/deepfool.py which should be commented out.
                # Otherwise, all logging calls are printed to the terminal
                # noinspection PyCallingNonCallable
                _, clipped_adv_imgs, _ = attack(**attack_kwargs)(
                    foolbox_model, batch_input, batch_criteria, epsilons=0.01,
                )
                img_outputs.append(clipped_adv_imgs)

            clipped_adv_imgs = torch.cat(img_outputs, dim=0)
            np.savez_compressed(previous_adv_output_path,
                                clipped_adv_imgs=clipped_adv_imgs.numpy(force=True),
                                original_imgs=self.exp.input.numpy(force=True))
            logger.debug(f"Saved generated adversarial images to {previous_adv_output_path}.")

        # noinspection PyUnboundLocalVariable
        adv_preds = self.run_model(clipped_adv_imgs).argmax(1)
        logger.debug(f"{len(clipped_adv_imgs)} adversarial examples (og->adv): "
                     f"{original_preds}->{adv_preds}.")
        # futurenote: adv examples often appear to be in 'loops' like in paper:
        #  "Counterfactual Explanations for Remote Sensing Time Series Data:
        #   An Application to Land Cover Classification"

        # Only compare similarity of explanations where the model predictions changed
        # since the explanations should reflect the underlying model (and change a lot)
        # noinspection PyTypeChecker
        success: np.ndarray[bool] = original_preds != adv_preds
        if not success.all():
            failed_idxs = np.flatnonzero(~success)
            logger.warning(f"Failed to successfully load/generate truly adversarial images for "
                           f"idxs={failed_idxs} ({len(failed_idxs)}/{len(success)}).")

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
