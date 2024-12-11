import numpy as np
import torch


def during_training_validation_step(model_to_validate, eval_criterion, val_data_iterator, train_loss_arr,
                                    train_acc_arr, step_num, epoch, device, dataloader_len):
    model_to_validate.eval()
    with torch.no_grad():
        print(f"Epoch {epoch:03} - Batch num {step_num:05}")
        val_loss_arr = np.zeros(0)
        val_acc_arr = np.zeros(0)
        for _ in range(20):  # iterate over 20 batches
            val_data = next(val_data_iterator)
            val_images = val_data["image"].to(device)
            val_labels: torch.Tensor = val_data["label"].to(device)

            val_predictions: torch.Tensor = model_to_validate(val_images)

            val_loss = eval_criterion(val_predictions, val_labels)
            val_loss_arr = np.append(val_loss_arr, val_loss.item())

            val_accuracy = (val_predictions.argmax(dim=1) == val_labels).float().mean().item()
            val_acc_arr = np.append(val_acc_arr, val_accuracy)

        print(f"Training loss: {train_loss_arr.mean():.2f}, Training accuracy: {train_acc_arr.mean():.2f}")
        print(f"Validation loss: {val_loss_arr.mean():.2f}, Validation accuracy: {val_acc_arr.mean():.2f}")

        log_dict = {
            "overall_step": epoch * dataloader_len + step_num,
            "training/loss": train_loss_arr.mean(),
            "training/accuracy": train_acc_arr.mean(),
            "validation/loss": val_loss_arr.mean(),
            "validation/accuracy": val_acc_arr.mean()
        }

        train_loss_arr = np.zeros(0)
        train_acc_arr = np.zeros(0)

        return log_dict


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
