from torch import nn, optim, FloatTensor, save, no_grad
from sklearn.metrics import accuracy_score
import joblib
from output.config import DATA_PATH

dataset = joblib.load(DATA_PATH)
input_size = dataset["X_train"].shape[1]

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.first_filter = nn.ReLU()
        self.hidden_layer = nn.Linear(128, 16)
        self.second_filter = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)
        self.last_filter = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.first_filter(x)
        x = self.hidden_layer(x)
        x = self.second_filter(x)
        x = self.layer2(x)
        x = self.last_filter(x)
        return x
    
def main():

    X_train_array = dataset["X_train"].toarray()
    X_test_array = dataset["X_test"].toarray()

    X_train = FloatTensor(X_train_array)
    X_test = FloatTensor(X_test_array)
    y_train = FloatTensor(dataset["y_train"].values.copy()).view(-1, 1)
    y_test = FloatTensor(dataset["y_test"].values.copy()).view(-1, 1) 

    model = Model()

    loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 0.001)

    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model.forward(X_train)
        output = loss(predictions, y_train)
        output.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(output.item())

    save(model.state_dict(), "output/model.pth")
    print("Training done, model saved")

    with no_grad():
        test_predictions = model(X_test)
        test_predictions_labels = (test_predictions > 0.5).float()

        acc = accuracy_score(y_test, test_predictions_labels)
        print(f"Test set accuracy : {acc * 100:.2f}%")

if __name__ == "__main__":
    main()