from Model import *

print("Loading Data")
ds = DataSet()
X, y, classes, labels, idx, labels_r, X_train, y_train, X_test, y_test = ds.load_data()
print(len(X_train), len(X_test), len(y_train), len(y_test))
print("Loaded Data")
print("Creating Model")
model = CNN().to(DEVICE)
criterion = MSELoss()
optimizer = Adam(model.parameter(), lr=0.001)
print("Created Model")
print("Training Model")
hp = Help_Funcs()
hp.train(
    "BaseLine",
    EPOCHS,
    X_train,
    y_train,
    X_test,
    y_test,
    BATCH_SIZE,
    DEVICE,
    model,
    criterion,
    optimizer,
)
print("Trained Model")
