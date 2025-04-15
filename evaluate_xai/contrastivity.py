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
            img_bound_quantile: float = 0.02,
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
        # Adversarial images are specific to a trained model (using their gradients) not an explainer
        # e.g. xai_output_unix/EuroSATRGB/c00/combined/ConvNeXtSmall_adversarial_examples.npz
        previous_adv_output_path = (self.exp.save_path.parent /  # not explainer specific so use .parent
                                    f"{self.exp.model.__class__.__name__}_adversarial_examples.npz")

        if previous_adv_output_path.exists():
            logger.debug(f"Adversarial images potentially already generated, loading from {previous_adv_output_path}.")
            with np.load(previous_adv_output_path) as data:
                clipped_adv_imgs = torch.from_numpy(data["clipped_adv_imgs"]).to(self.exp.device)
                original_imgs = torch.from_numpy(data["original_imgs"]).to(self.exp.device)

            # check if previous generation was for different number of inputs/sample size
            if original_imgs.shape != self.exp.input.shape:
                num_desired = self.n_samples
            else:  # if shapes are the same, we check all images
                num_desired = original_imgs.shape[0]

            if torch.equal(original_imgs[:num_desired], self.exp.input):
                logger.debug(f"Saved adversarial images ({num_desired}/{original_imgs.shape[0]}) "
                             f"are for current explanation input. Loading these and skipping generation.")
                clipped_adv_imgs = clipped_adv_imgs[:num_desired]  # only need the num_desired for this run
                need_to_generate = False  # no need to generate after all!
            else:
                logger.warning(f"Saved adversarial images (shape={original_imgs.shape}) were not for "
                               f"current explanation input (shape={self.exp.input.shape}).")

        # if no existing adversarial images, generate them
        if need_to_generate:
            img_outputs = []
            attack_batch_size = self.max_batch_size
            if attack_batch_size == 0:
                attack_batch_size = self.n_samples

            logger.info(f"No existing adversarial images at {previous_adv_output_path}. "
                        f"Generating new ones via foolbox with batch size {attack_batch_size}.")

            expected_img_bounds = (-1, 1)
            attack_epsilon = 0.01  # default good for the vast majority of datasets and attacks
            higher_epsilon = 0.05  # higher epsilon to account for the larger image value range
            # Images may be out of bounds due to type of normalisation (especially for MS data)
            input_min = self.exp.input.min()
            input_max = self.exp.input.max()
            if input_min < expected_img_bounds[0] or input_max > expected_img_bounds[1]:
                logger.warning(f"Input images were outside the expected bounds {expected_img_bounds}: "
                               f"input.min()={input_min}, input.max()={input_max}. "
                               f"Updating image bounds before generation.")

                # PyTorch's quantile function has an arbitrary input size limit of 16M elements so
                # we need to use numpy's quantile function instead
                # See https://github.com/pytorch/pytorch/issues/64947
                input_lq = np.quantile(self.exp.input.numpy(force=True), img_bound_quantile)
                input_uq = np.quantile(self.exp.input.numpy(force=True), 1-img_bound_quantile)
                if input_lq < expected_img_bounds[0] or input_uq > expected_img_bounds[1]:
                    logger.warning(f"Input images were *significantly* ({img_bound_quantile*100:.1f}% quantiles) "
                                   f"outside the expected bounds {expected_img_bounds}: "
                                   f"input.quantile({img_bound_quantile})={input_lq}, "
                                   f"input.quantile({1-img_bound_quantile})={input_uq}. "
                                   f"Using higher epsilon "
                                   f"({attack_epsilon} -> {higher_epsilon}).")
                    attack_epsilon = higher_epsilon

                expected_img_bounds = (min(input_min, expected_img_bounds[0]),
                                       max(input_max, expected_img_bounds[1]))

            foolbox_model = foolbox.PyTorchModel(self.exp.model, device=self.exp.device, bounds=expected_img_bounds)

            for batch_input, batch_preds in zip(
                    helpers.utils.make_device_batches(
                        self.exp.input, attack_batch_size, self.exp.device),
                    helpers.utils.make_device_batches(
                        torch.from_numpy(original_preds), attack_batch_size, self.exp.device),
            ):
                # We don't care about the images' true labels,
                # just the model's original predictions (and associated explanation)
                batch_criteria = foolbox.criteria.Misclassification(batch_preds)

                # WARNING: there is a rogue call to the base logging.info() on
                # line 127 of foolbox/attacks/deepfool.py which should be commented out.
                # Otherwise, all logging calls are printed to the terminal
                # noinspection PyCallingNonCallable
                _, clipped_adv_imgs, _ = attack(**attack_kwargs)(
                    foolbox_model, batch_input, batch_criteria, epsilons=attack_epsilon,
                )
                img_outputs.append(clipped_adv_imgs)

            clipped_adv_imgs = torch.cat(img_outputs, dim=0)

            if not previous_adv_output_path.exists():  # create the directory if it doesn't exist
                previous_adv_output_path.parent.mkdir(parents=True, exist_ok=True)

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
