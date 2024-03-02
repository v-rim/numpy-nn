def main():
    import numpy as np
    from model import Model
    from layers import FullyConnected, ReLU, Tanh
    from loss import MSELoss
    from dataloader import CSVDataLoader
    import matplotlib.pyplot as plt

    np.set_printoptions(formatter={"float": "{:0.4f}".format})

    # Define the model
    layers = [
        FullyConnected(6, 32),
        ReLU(),
        FullyConnected(32, 32),
        ReLU(),
        FullyConnected(32, 32),
        ReLU(),
        FullyConnected(32, 3),
        Tanh(),
    ]
    loss = MSELoss()
    learning_rate = 0.1
    model = Model(layers, loss, learning_rate)

    # Get training and testing sets
    dataloader = CSVDataLoader(
        "dataset_2.csv",
        set_ratios=[7, 3, 15], # Randomly throw out some data
        batch_size=16,
        shuffle=True,
    )
    training_set = dataloader.get_set(0)
    testing_set = dataloader.get_set(1)

    # Combine test_set into a single array
    testing_set = np.concatenate(testing_set, axis=0)

    # Train and test
    epochs = 5
    training_loss = []
    testing_loss = []
    for i in range(epochs):
        learning_rate *= 0.75 # Decay learning rate for higher epochs
        for j, batch in enumerate(training_set):
            X = batch[:, :6]
            y = batch[:, 6:]
            data_loss = model.train(X, y)

            # Get training loss and test loss every 5 batches
            if j % 5 == 0:
                training_loss.append(data_loss)

                X = testing_set[:, :6]
                y = testing_set[:, 6:]
                y_pred = model.predict(X)
                data_loss = loss.forward(y_pred, y)
                testing_loss.append(data_loss)
            if j % 100 == 0:
                print(f"Epoch: {i} | Batch: {j} | Loss: {data_loss:0.6f}")

    plt.plot(range(len(training_loss)), training_loss, testing_loss)
    plt.show()


if __name__ == "__main__":
    main()
