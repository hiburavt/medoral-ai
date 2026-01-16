import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
DATA_DIR = "dataset/processed"
MODEL_SAVE_PATH = "model/oral_cancer_model.h5"

def build_model():
    """
    Builds the model using ResNet50 as base.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model

def train():
    if not os.path.exists(os.path.join(DATA_DIR, "train")):
        print(f"Error: Training data not found at {DATA_DIR}/train. Run scripts/prepare_data.py first.")
        return

    # Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Loading data...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    # Build Model
    model, base_model = build_model()
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train Phase 1 (Frozen Base)
    print("\nStarting Phase 1 Training (Frozen Base)...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    # Fine-tuning (Optional but recommended)
    print("\nStarting Phase 2: Fine-tuning...")
    base_model.trainable = True
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE/10), # Lower learning rate
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
                  
    history_fine = model.fit(
        train_generator,
        epochs=10, # Fewer epochs for fine-tuning
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
