import json, tensorflow as tf, os

H5_PATH = "best_b7_model.h5" 
LABELS_OUTPUT = "labels.json"
TFLITE_OUTPUT = "skin_model_int8.tflite"

if not os.path.exists(H5_PATH):
    raise FileNotFoundError(f"Файл не найден: {H5_PATH}")

model = tf.keras.models.load_model(H5_PATH)
label_map = {0: "Доброкачественное", 1: "Меланома"}

with open(LABELS_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
tflite_model = converter.convert()

with open(TFLITE_OUTPUT, "wb") as f:
    f.write(tflite_model)

print("Гтоово")