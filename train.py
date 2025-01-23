import os
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from utils import load_data, splitdata, create_model

# Load dataset
print("Loading dataset...")
X, y = load_data()


# Split dataset into training, validation, and test sets
print("Splitting dataset...")
data = splitdata(X, y, test_size=0.1, valid_size=0.1)

# Create the model
print("Creating the model...")
model = create_model()

# Define callbacks
log_dir = "logs"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# use tensorboard to view metrics
tensorboard = TensorBoard(log_dir=log_dir)
# define early stopping to stop training after 10 epochs of not improving
early_stopping = EarlyStopping(mode="min", patience=10, restore_best_weights=True)

# Training parameters
batch_size = 64
epochs = 100

# Train the model
print("Starting training...")
history = model.fit(
    data["X_train"],
    data["y_train"],
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(data["X_valid"], data["y_valid"]),
    callbacks=[tensorboard, early_stopping]
)

# Save the trained model
model_save_path = "results/model.h5"
if not os.path.exists("results"):
    os.mkdir("results")

model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Evaluate the model
print(f"Evaluating the model using {len(data['X_test'])} samples...")
loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")
