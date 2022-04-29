from Model import *

print("Loading Data")
ds = DataSet()
X, y, classes, labels, idx, labels_r, X_train, y_train, X_test, y_test = ds.load_data()
print(len(X_train), len(X_test), len(y_train), len(y_test))
print("Loaded Data")
print("Creating Model")
model = CNN(idx_of_classes=idx).to(DEVICE)
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)
print("Created Model")
print("Training Model")
hp = Help_Funcs()
hp.train(
    PROJECT_NAME,
    "BaseLine CNN",
    EPOCHS,
    X_train.view(-1, 1, 28, 28).to(DEVICE),
    y_train.to(DEVICE),
    X_test.view(-1, 1, 28, 28),
    y_test.to(DEVICE),
    BATCH_SIZE,
    DEVICE,
    model,
    criterion,
    optimizer,
    labels_r,
)
print("Trained Model")
