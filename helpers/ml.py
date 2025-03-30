import typing as t

import numpy as np
import torch
from tqdm.autonotebook import tqdm

from helpers import utils


def make_preds(
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: torch.nn.Module,
) -> t.Tuple[torch.Tensor, torch.Tensor]:

    y_pred: torch.Tensor = model(x)

    loss = criterion(y_pred, y)
    accuracy = (y_pred.argmax(dim=1) == y).float().mean()

    return loss, accuracy


def train_step(
        model_to_train: torch.nn.Module,
        input_img: torch.Tensor,
        targets: torch.Tensor,
        train_criterion: torch.nn.Module,
        model_optimiser: torch.optim.Optimizer,
) -> t.Tuple[float, float]:

    model_to_train.train()
    model_device = utils.get_model_device(model_to_train)

    # use set_to_none=True to save some memory
    # see caveats at https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
    model_optimiser.zero_grad(set_to_none=True)
    loss, accuracy = make_preds(model_to_train, input_img.to(model_device), targets.to(model_device), train_criterion)

    loss.backward()
    model_optimiser.step()

    return loss.item(), accuracy.item()


def validation_step(
        model_to_validate: torch.nn.Module,
        eval_criterion: torch.nn.Module,
        val_data_gen: t.Generator,
        num_val_batches: int,
) -> t.Tuple[float, float]:

    model_to_validate.eval()
    model_device = utils.get_model_device(model_to_validate)

    val_loss_arr = np.zeros(0)
    val_acc_arr = np.zeros(0)

    with torch.no_grad():
        for _ in tqdm(range(num_val_batches), desc="Validating", unit="batch", ncols=110, leave=False):
            val_data = next(val_data_gen)
            val_images = val_data["image"].to(model_device)
            val_labels = val_data["label"].to(model_device)

            loss, accuracy = make_preds(model_to_validate, val_images, val_labels, eval_criterion)

            val_loss_arr = np.append(val_loss_arr, loss.item())
            val_acc_arr = np.append(val_acc_arr, accuracy.item())

    # futuretodo: return some examples of misclassified images?
    return val_loss_arr.mean(), val_acc_arr.mean()


def sample_outputs(
        model_to_test: torch.nn.Module,
        sample_iterator: t.Generator[dict[str, torch.Tensor], None, None],
        num_batches: int,
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    model_to_test.eval()
    model_device = utils.get_model_device(model_to_test)

    with torch.no_grad():
        all_samples = torch.zeros(0).to(model_device)
        all_labels = torch.zeros(0).to(model_device)
        all_outputs = torch.zeros(0).to(model_device)

        for _ in tqdm(range(num_batches), desc="Sampling", unit="batch", ncols=110, leave=False):
            data = next(sample_iterator)
            images = data["image"].to(model_device)
            labels = data["label"].to(model_device)

            all_samples = torch.cat([all_samples, images], dim=0)
            all_labels = torch.cat([all_labels, labels], dim=0)

            output = model_to_test(images)
            all_outputs = torch.cat([all_outputs, output], dim=0)

    return all_samples.cpu(), all_labels.cpu().int(), all_outputs.cpu()


def test_model(model_to_test, testing_dataloader, device):
    model_to_test.eval()
    with torch.no_grad():
        batch_accuracy = np.zeros(0)

        all_labels = np.zeros(0)
        all_predictions = np.zeros(0)
        all_logits = np.zeros(0)

        for i, data in enumerate(testing_dataloader):
            print(f"Assessing batch {i:02}/{len(testing_dataloader)}", end="\r")
            images = data["image"].to(device)
            labels = data["label"].to(device)

            logits = model_to_test(images)
            all_logits = np.append(all_logits, logits.cpu().numpy(),
                                   axis=0) if all_logits.size else logits.cpu().numpy()
            predictions = logits.argmax(dim=1, keepdim=True)

            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_predictions = np.append(all_predictions, predictions.cpu().numpy())

            batch_accuracy = np.append(batch_accuracy,
                                       predictions.data.eq(labels.view_as(predictions)).float().mean().item())

        print(f"Accuracy of the model on the {len(all_labels)} test images: {batch_accuracy.mean():.5f}")

    return all_labels, all_predictions, all_logits
