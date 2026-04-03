import os, pandas as pd, tensorflow as tf, glob
from tensorflow.keras import layers, models, applications, optimizers

IMG_SIZE, BATCH_SIZE, EPOCHS, LR = (600, 600), 4, 10, 1e-5

PATH_ISIC = "/kaggle/input/skin-cancer9-classesisic/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
PATH_AUG = "/kaggle/input/datasets/valdies/augment/melanoma_augmented"
PATH_HAM_META = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"
PATH_HAM_IMG1 = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1"
PATH_HAM_IMG2 = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2"

def collect_all_data():
    paths, labels = [], []
    if os.path.exists(PATH_ISIC):
        for folder in os.listdir(PATH_ISIC):
            folder_path = os.path.join(PATH_ISIC, folder)
            if not os.path.isdir(folder_path): continue
            label = 1 if "melanoma" in folder.lower() else 0
            imgs = glob.glob(os.path.join(folder_path, "*.jpg"))
            paths.extend(imgs); labels.extend([label] * len(imgs))
    if os.path.exists(PATH_HAM_META):
        df = pd.read_csv(PATH_HAM_META)
        for _, row in df.iterrows():
            img_id = row['image_id'] + ".jpg"
            p1, p2 = os.path.join(PATH_HAM_IMG1, img_id), os.path.join(PATH_HAM_IMG2, img_id)
            target = 1 if row['dx'] == 'mel' else 0
            if os.path.exists(p1): paths.append(p1); labels.append(target)
            elif os.path.exists(p2): paths.append(p2); labels.append(target)
    if os.path.exists(PATH_AUG):
        aug_imgs = glob.glob(os.path.join(PATH_AUG, "*.jpg"))
        paths.extend(aug_imgs); labels.extend([1] * len(aug_imgs))
    return paths, labels

all_paths, all_labels = collect_all_data()

def parse_function(filename, label):
    img = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
    return tf.image.resize(img, IMG_SIZE), label

ds = tf.data.Dataset.from_tensor_slices((all_paths, all_labels)).shuffle(len(all_paths))
ds = ds.map(parse_function, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1)

base = applications.EfficientNetB7(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = True

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizers.Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])

class SimpleLog(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nЭпоха {epoch + 1}/{EPOCHS}")

checkpoint = tf.keras.callbacks.ModelCheckpoint("best_b7_model.h5", save_best_only=True, monitor='accuracy')
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2)

model.fit(ds, epochs=EPOCHS, callbacks=[checkpoint, lr_reducer, SimpleLog()], verbose=1)
model.save("final_skin_model_b7.h5")