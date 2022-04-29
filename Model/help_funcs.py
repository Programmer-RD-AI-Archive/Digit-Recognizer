"""sumary_line"""
from Model import *
from Model.metrics import *


class Help_Funcs:
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
    ):
        m = Metrics()
        wandb.init(project=PROJECT_NAME, name=name)
        for _ in tqdm(range(epochs)):
            for idx in range(0, len(X_train), batch_size):
                X_batch = X_train[idx: idx + batch_size].to(device)
                y_batch = y_train[idx: idx + batch_size].to(device)
                preds = model(X_batch)
                loss = criterion(preds)
                optimizer.step()
                loss.backward()
                optimizer.zero_grad()
            wandb.log(
                {
                    "Accuracy Batch": m.accuracy(model, X_batch, y_batch),
                    "Loss Batch": loss.item(),
                    "Accuracy": m.accuracy(model, X_test, y_test),
                    "Loss": m.loss(model, X_test, y_test, criterion),
                }
            )
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
