import csv

# Define CSV file and column headers
csv_file = "multiple_runs_metrics.csv"
fieldnames = ["run", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "test_loss", "test_accuracy"]

# Initialize the CSV file with headers
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

# Loop for multiple runs
for run in range(1, num_runs + 1):
    print(f"Running model training {run}")
    
    # Train the model
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(valX, valY), verbose=0)

    # Evaluate on test data
    test_metrics = model.evaluate(testX, testY, verbose=0)
    test_metrics = dict(zip(model.metrics_names, test_metrics))
    
    # Get training and validation metrics from the last epoch
    train_loss = history.history["loss"][-1]
    train_accuracy = history.history["accuracy"][-1]
    val_loss = history.history["val_loss"][-1]
    val_accuracy = history.history["val_accuracy"][-1]
    test_loss = test_metrics["loss"]
    test_accuracy = test_metrics["accuracy"]
    
    # Append metrics to CSV file
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            "run": run,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })
