import typing

import numpy as np
import torch
from tqdm.autonotebook import tqdm

import utils


def make_preds(
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: torch.nn.Module,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:

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
) -> typing.Tuple[float, float]:

    model_to_train.train()
    model_device = utils.get_model_device(model_to_train)

    loss, accuracy = make_preds(model_to_train, input_img.to(model_device), targets.to(model_device), train_criterion)

    model_optimiser.zero_grad()
    loss.backward()
    model_optimiser.step()

    return loss.item(), accuracy.item()


def validation_step(
        model_to_validate: torch.nn.Module,
        eval_criterion: torch.nn.Module,
        val_data_gen: typing.Generator,
        num_val_batches: int,
) -> typing.Tuple[float, float]:

    model_to_validate.eval()
    model_device = utils.get_model_device(model_to_validate)

    val_loss_arr = np.zeros(0)
    val_acc_arr = np.zeros(0)

    with torch.no_grad():
        for _ in tqdm(range(num_val_batches), desc="Validating", ncols=110, leave=False):
            val_data = next(val_data_gen)
            val_images = val_data["image"].to(model_device)
            val_labels = val_data["label"].to(model_device)

            loss, accuracy = make_preds(model_to_validate, val_images, val_labels, eval_criterion)

            val_loss_arr = np.append(val_loss_arr, loss.item())
            val_acc_arr = np.append(val_acc_arr, accuracy.item())

    return val_loss_arr.mean(), val_acc_arr.mean()


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
