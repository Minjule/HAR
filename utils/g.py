import os
import numpy as np
import tensorflow as tf
from keras import layers, models

# --------------------------
# 1. Load SPHAR data
# --------------------------
def load_sphar_data(data_dir):
    """
    Assumes SPHAR dataset is structured like:
    data_dir/
        action_class_1/
            sample1.npy
            sample2.npy
        action_class_2/
            ...
    Each .npy contains shape (T, J, 3) where
    T = frames, J = joints, 3 = x,y,z coords
    """
    X, y = [], []
    class_names = sorted(os.listdir(data_dir))
    label_map = {cls: i for i, cls in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for f in os.listdir(cls_dir):
            if f.endswith(".npy"):
                arr = np.load(os.path.join(cls_dir, f), allow_pickle=True)  # shape (T, J, 3)
                X.append(arr)
                y.append(label_map[cls])
    
    return X, np.array(y), class_names

# --------------------------
# 2. Pad / normalize
# --------------------------
def preprocess_sequences(X, max_len=100):
    """
    Pad or truncate sequences to max_len frames.
    Flatten joint coords into (T, J*3).
    """
    processed = []
    for seq in X:
        T, J, C = seq.shape
        seq = seq.reshape(T, J*C)  # flatten joints
        if T < max_len:
            pad = np.zeros((max_len - T, J*C))
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:max_len]
        processed.append(seq)
    return np.array(processed, dtype=np.float32)

# --------------------------
# 3. Build model
# --------------------------
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Masking(mask_value=0.0, input_shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# --------------------------
# 4. Run training
# --------------------------
if __name__ == "__main__":
    data_dir = "C:\\Users\\Acer\\Desktop\\HAR\\SPHAR-Dataset-1.0" 
    X, y, class_names = load_sphar_data(data_dir)

    # Preprocess
    X_proc = preprocess_sequences(X, max_len=100)
    print("Data shape:", X_proc.shape, "Labels:", y.shape)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42
    )

    # Build + train
    model = build_model(input_shape=X_train.shape[1:], num_classes=len(class_names))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)
