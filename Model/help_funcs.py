"""sumary_line"""
from Model import *
from Model.metrics import *


class Help_Funcs:
    @staticmethod
    def train(
        PROJECT_NAME,
        name,
        epochs,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size,
        device,
        model,
        criterion,
        optimizer,
        labels_r,
    ):
        m = Metrics()
        wandb.init(
            project=PROJECT_NAME,
            name=name,
            config={"device": device,
                    "batch_size": batch_size, "epochs": epochs},
        )
        wandb.watch(model)
        for _ in tqdm(range(epochs)):
            for idx in range(0, len(X_train), batch_size):
                X_batch = X_train[idx: idx + batch_size].float().to(device)
                y_batch = y_train[idx: idx + batch_size].float().to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            wandb.log(
                {
                    "Accuracy Batch": m.accuracy(model, X_batch, y_batch),
                    "Loss Batch": loss.item(),
                    "Accuracy": m.accuracy(
                        model.to(device),
                        X_test.to(device).float(),
                        y_test.to(device).float(),
                    ),
                    "Loss": m.loss(
                        model.to(device),
                        X_test.to(device).float(),
                        y_test.to(device).float(),
                        criterion,
                    ),
                }
            )
            model.train()
        # preds_imgs = m.test_images(model, labels_r, device)
        # for pred_img in preds_imgs:
        #     wandb.log({pred_img[0]: wandb.Image(pred_img[1])})
        wandb.watch(model)
        wandb.finish()
        return (
            PROJECT_NAME,
            name,
            epochs,
            X_train,
            y_train,
            X_test,
            y_test,
            batch_size,
            device,
            model,
            criterion,
            optimizer,
        )
