import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "dataset/processed"
MODEL_PATH = "model/oral_cancer_model.h5"

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Train the model first.")
        return

    if not os.path.exists(os.path.join(DATA_DIR, "test")):
        print(f"Error: Test data not found at {DATA_DIR}/test.")
        return

    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Loading test data...")
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False # Important for correct evaluation
    )
    
    print("Evaluating...")
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int).reshape(-1)
    y_true = test_generator.classes
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Suspicious']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Healthy', 'Suspicious'], rotation=45)
    plt.yticks(tick_marks, ['Healthy', 'Suspicious'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("Saved roc_curve.png")

if __name__ == "__main__":
    evaluate()
