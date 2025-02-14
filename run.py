
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import os

original_data_dir = "C:/Users/21wat/Documents/Python ML/Projects/MLProject4/6 Emotions for image classification"
save_dir = "saved_models"

# Parameters
img_size = (128, 128)
batch_size = 32
epochs = 5
epoch_steps = 5

# Image dataset from directory
def create_datasets(dir):
    training_set = tf.keras.utils.image_dataset_from_directory(
        dir,
        validation_split=0.2,
        subset="training",
        label_mode="categorical",
        seed=0,
        image_size=img_size
    )

    validation_set = tf.keras.utils.image_dataset_from_directory(
        dir,
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        seed=0,
        image_size=img_size
    )

    return training_set, validation_set

# 1. CNN without pooling or dropout
def build_cnn_no_pooling(train_set):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(train_set.class_names), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 2. CNN with pooling and dropout
def build_cnn_with_pooling(train_set):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(train_set.class_names), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to train and save models
def train_and_save_model(model, model_name):
    checkpoint_path = os.path.join(save_dir, model_name + '.keras')  # Change .h5 to .keras
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')

    history = model.fit(train_set,
                        validation_data=val_set,
                        epochs=epochs,
                        callbacks=[checkpoint])
    return history, checkpoint_path


# Fine-tuning a pre-trained model (Xception)
def build_pretrained_xception(train_set):
    base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(train_set.class_names), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

train_set, val_set = create_datasets(original_data_dir)
noPooldir = save_dir + '/cnn_no_pooling.keras'
pooldir = save_dir + '/cnn_with_pooling.keras'
xceptiondir = save_dir + '/xception_finetuned.keras'

# Build the models
if os.path.exists(noPooldir):
    cnn_no_pooling = tf.keras.models.load_model(noPooldir)
else:
    print('NEW')
    cnn_no_pooling = build_cnn_no_pooling(train_set)
    
if os.path.exists(pooldir):
    cnn_with_pooling = tf.keras.models.load_model(pooldir)
else:
    print('NEW')
    cnn_with_pooling = build_cnn_with_pooling(train_set)

if os.path.exists(xceptiondir):
    xception_model = tf.keras.models.load_model(xceptiondir)
else:
    print('NEW')
    xception_model = build_pretrained_xception(train_set)

results = {
    "cnn_no_pooling": {"loss": [], "accuracy": []},
    "cnn_with_pooling": {"loss": [], "accuracy": []},
    "xception_finetuned": {"loss": [], "accuracy": []}
}

# Evaluate models
def evaluate_model(model_name, model_path, i):
    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(val_set)
    results[model_name]["loss"].append(loss)
    results[model_name]["accuracy"].append(accuracy)
    print(f"Step {i+1} - Model: {model_name}, Loss: {loss}, Accuracy: {accuracy}")
    # return loss, accuracy

cnn_no_pooling.summary()
cnn_with_pooling.summary()
xception_model.summary()

# Train the models
for i in range(epoch_steps):
    _, cnn_no_pooling_path = train_and_save_model(cnn_no_pooling, "cnn_no_pooling")
    _, cnn_with_pooling_path = train_and_save_model(cnn_with_pooling, "cnn_with_pooling")
    _, xception_path = train_and_save_model(xception_model, "xception_finetuned")



    evaluate_model('cnn_no_pooling' , cnn_no_pooling_path, i)
    evaluate_model('cnn_with_pooling', cnn_with_pooling_path, i)
    evaluate_model('xception_finetuned', xception_path, i)
    
    # loss, accuracy = evaluate_model(model_path)

# Create a DataFrame from results
table_data = []
for model_name in results.keys():
    for j in range(len(results[model_name]["loss"])):
        table_data.append({
            "Epoch Step": f"{(j+1) * epochs}",
            "Model": model_name,
            "Loss": results[model_name]["loss"][j],
            "Accuracy": results[model_name]["accuracy"][j]
        })

# Display table
results_df = pd.DataFrame(table_data)
print("\nResults Table:")
print(results_df.to_string(index=False))


# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Loss plot
for model_name in results.keys():
    axes[0].plot(range(epochs), results[model_name]["loss"], label=model_name)
axes[0].set_title("Loss vs Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Accuracy plot
for model_name in results.keys():
    axes[1].plot(range(epochs), results[model_name]["accuracy"], label=model_name)
axes[1].set_title("Accuracy vs Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.show()