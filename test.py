import os
import io
import random
import nibabel as nib
import numpy as np
from skimage.transform import resize
from itertools import combinations
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from sklearn.model_selection import train_test_split
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam as TFAdam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence


def dice_coef(y_true, y_pred, smooth=1.):
  y_true_f = tf.keras.backend.flatten(y_true)
  y_pred_f = tf.keras.backend.flatten(y_pred)
  intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def tpr(y_true, y_pred, threshold=0.5):
    y_pred_pos = tf.cast(y_pred > threshold, tf.float32)
    y_true_pos = tf.cast(y_true > threshold, tf.float32)
    
    true_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_pos == 1, y_pred_pos == 1), tf.float32))
    actual_pos = tf.reduce_sum(tf.cast(y_true_pos, tf.float32))
    
    tpr = true_pos / (actual_pos + tf.keras.backend.epsilon())
    return tpr

def fpr(y_true, y_pred, threshold=0.5):
    y_pred_pos = tf.cast(y_pred > threshold, tf.float32)
    y_true_neg = tf.cast(y_true <= threshold, tf.float32)
    
    false_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_neg == 1, y_pred_pos == 1), tf.float32))
    actual_neg = tf.reduce_sum(tf.cast(y_true_neg, tf.float32))
    
    fpr = false_pos / (actual_neg + tf.keras.backend.epsilon())
    return fpr

def pad_z_dimension(image, desired_z_size=64, pad_value=0):
    current_z_size = image.shape[2]

    if current_z_size >= desired_z_size:
        start_idx = (current_z_size - desired_z_size) // 2
        end_idx = start_idx + desired_z_size
        return image[:, :, start_idx:end_idx]
    else:
        total_pad = desired_z_size - current_z_size
        pad_front = total_pad // 2
        pad_back = total_pad // 2

        if total_pad % 2 != 0:
            pad_back += 1

        padded_image = np.pad(image, ((0, 0), (0, 0), (pad_front, pad_back)), 'constant', constant_values=pad_value)
        return padded_image


def normalize_image(image):
    image = image - np.min(image)
    image = image / np.max(image)
    return image

def generate_all_masks(num_modalities=5):
    all_masks = []

    for n in range(1, num_modalities + 1):
        for indices in combinations(range(num_modalities), n):
            mask = np.zeros(num_modalities, dtype=int)
            mask[list(indices)] = 1 
            all_masks.append(mask)

    return all_masks

class NiiDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_folders, mask_folders, batch_size, img_size=(128, 128), z_size=64, indices=None, num_modalities=5):
        self.image_folders = image_folders
        self.mask_folders = mask_folders
        self.batch_size = batch_size
        self.img_size = img_size
        self.z_size = z_size
        self.num_modalities = num_modalities
        self.all_masks = generate_all_masks(num_modalities)  # Precompute all mask combinations
        self.mask_count = len(self.all_masks)
        
        # Get all file lists
        self.image_files = [sorted(os.listdir(folder)) for folder in image_folders]
        self.mask_files = [sorted(os.listdir(folder)) for folder in mask_folders]

        # Calculate file counts for looping logic
        self.max_files_per_modality = max(len(files) for files in self.image_files)
        self.image_file_counts = [len(files) for files in self.image_files]
        self.mask_file_counts = [len(files) for files in self.mask_files]

        # Use custom indices or default to all valid indices
        self.indices = indices if indices is not None else list(range(self.max_files_per_modality))

        # Mask combination tracking
        self.current_mask_index = 0  # Start at the first mask combination
        self.used_masks = list(range(self.mask_count))  # Cycle through all mask indices
    
    def __len__(self):
        # Number of batches per epoch, accounting for all masks
        return int(np.ceil(len(self.indices)/ self.batch_size))
    
    def __getitem__(self, index):
        batch_images, batch_masks = [], []

        # Randomly select one mask combination for this epoch
        if index == 0 and self.current_mask_index == 0:  # Select a new mask at the start of an epoch
            self.selected_mask = random.choice(self.all_masks)

        # Determine start and end indices for this batch
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices))

        for i in range(start_idx, end_idx):
            volume_idx = self.indices[i]  # Use sequential index for data selection
            slice_idx = random.randint(0, self.z_size - 1)  # Pick a random slice for this volume
            
            # Load data for the selected slice and mask
            stacked_images, stacked_masks = self.load_data(volume_idx, slice_idx, self.selected_mask)
            batch_images.append(stacked_images)
            batch_masks.append(stacked_masks)

        # Convert to numpy arrays
        return np.array(batch_images, dtype=np.float32), np.array(batch_masks, dtype=np.float32)


    def _get_repeated_index(self, modality_idx, volume_idx):
        """Get a valid file index for the modality, looping back if needed."""
        file_count = self.image_file_counts[modality_idx]
        return volume_idx % file_count  # Loop back when index exceeds file count

    def load_data(self, volume_idx, slice_idx, mask):
        stacked_images = []
        stacked_masks = []

        # Load images or use dummy channels for missing modalities
        for j in range(len(self.image_folders)):
            # Reuse indices when volume_idx exceeds modality file count
            image_idx = self._get_repeated_index(j, volume_idx)
            image_path = os.path.join(self.image_folders[j], self.image_files[j][image_idx])

            if image_path.endswith('.nii'):
                image = nib.load(image_path).get_fdata()
                image = pad_z_dimension(image, self.z_size, 0)
                image = resize(image, (*self.img_size, self.z_size))
                image = normalize_image(image)
            else:
                # Use a dummy channel if file does not exist
                image = np.zeros((self.img_size[0], self.img_size[1], self.z_size))

            # If modality is selected in the mask, add the slice; otherwise, add zeros
            if mask[j] == 1:
                stacked_images.append(image[:, :, slice_idx])
            else:
                stacked_images.append(np.zeros((self.img_size[0], self.img_size[1])))

        # Load masks or use dummy channels for missing modalities
        for j in range(len(self.mask_folders)):
            mask_idx = self._get_repeated_index(j, volume_idx)
            mask_path = os.path.join(self.mask_folders[j], self.mask_files[j][mask_idx])

            if mask_path.endswith('.nii'):
                mask_volume = nib.load(mask_path).get_fdata()
                mask_volume = pad_z_dimension(mask_volume, self.z_size, 0)
                mask_volume = resize(mask_volume, (*self.img_size, self.z_size))
                mask_volume = normalize_image(mask_volume)
            else:
                mask_volume = np.zeros((self.img_size[0], self.img_size[1], self.z_size))

            # If modality is selected in the mask, add the slice; otherwise, add zeros
            if mask[j] == 1:
                stacked_masks.append(mask_volume[:, :, slice_idx])
            else:
                stacked_masks.append(np.zeros((self.img_size[0], self.img_size[1])))

        # Stack all modalities for this slice
        stacked_images = np.stack(stacked_images, axis=-1)
        stacked_masks = np.stack(stacked_masks, axis=-1)

        return stacked_images, stacked_masks


def simple_unet_model(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_CHANNELS=5):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    #Contraction path
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(IMG_CHANNELS, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #optimizer = TFAdam(learning_rate=0.001)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    
    model.summary()
    
    return model

ff_folder = 'C:/Users/User/Desktop/multi-modal_data/ff/volumes/'
water_folder = 'C:/Users/User/Desktop/multi-modal_data/water/volumes/'
ff_r2_water_mask_folder = 'C:/Users/User/Desktop/multi-modal_data/ff/segmentations/'

ct_folder = 'C:/Users/User/Desktop/multi-modal_data/ct/volumes/'
ct_mask_folder = 'C:/Users/User/Desktop/multi-modal_data/ct/segmentations/'

mag_folder = 'C:/Users/User/Desktop/multi-modal_data/mag/volumes/'
mag_mask_folder = 'C:/Users/User/Desktop/multi-modal_data/mag/segmentations/'

pdff_folder = 'C:/Users/User/Desktop/multi-modal_data/pdff/volumes/'
pdff_mask_folder = 'C:/Users/User/Desktop/multi-modal_data/pdff/segmentations/'

axial_inphase_folder = 'C:/Users/User/Desktop/multi-modal_data/inphase/volumes/'
axial_inphase_mask_folder = 'C:/Users/User/Desktop/multi-modal_data/inphase/segmentations/'

axial_opposed_folder = 'C:/Users/User/Desktop/multi-modal_data/opposed/volumes/'
axial_opposed_mask_folder = 'C:/Users/User/Desktop/multi-modal_data/opposed/segmentations/'

portal_venous_folder = 'C:/Users/User/Desktop/multi-modal_data/portalvenous/volumes/'
portal_venous_mask_folder = 'C:/Users/User/Desktop/multi-modal_data/portalvenous/segmentations/'


all_image_folders = [ff_folder, pdff_folder, mag_folder, axial_inphase_folder, axial_opposed_folder]
all_mask_folders = [ff_r2_water_mask_folder, pdff_mask_folder, mag_mask_folder, axial_inphase_mask_folder, axial_opposed_mask_folder]

for folder in all_image_folders:
    print(f"{folder}: {len(os.listdir(folder))} volumes")

train_indices, val_indices = train_test_split(
    list(range(len(os.listdir(all_image_folders[0])))),
    test_size=0.2,  
    random_state=42
)

batch_size = 128

# Training generator
train_gen = NiiDataGenerator(
    image_folders=all_image_folders,
    mask_folders=all_mask_folders,
    batch_size=batch_size,
    indices=train_indices
)

# Ensure generator is compatible
x, y = train_gen[0]  # Test the generator
print(f"Batch X shape: {x.shape}, Batch Y shape: {y.shape}")
print(f"X dtype: {x.dtype}, Y dtype: {y.dtype}")

x, y = next(iter(train_gen))
print(f"Generator batch shapes - X: {x.shape}, Y: {y.shape}")
print(f"Generator batch types - X: {x.dtype}, Y: {y.dtype}")

# Validation generator
val_gen = NiiDataGenerator(
    image_folders=all_image_folders,
    mask_folders=all_mask_folders,
    batch_size=batch_size,
    indices=val_indices
)

print(f"Train generator length: {len(train_gen)}, Validation generator length: {len(val_gen)}")


def generator_to_tf_dataset(generator):
    output_signature = (
        tf.TensorSpec(shape=(None, 128, 128, 5), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 128, 128, 5), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(lambda: generator, output_signature=output_signature)

train_dataset = generator_to_tf_dataset(train_gen).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = generator_to_tf_dataset(val_gen).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

# Define and train the model
model = simple_unet_model(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_CHANNELS=5)
checkpoint = ModelCheckpoint('multimodal.h5', monitor='val_loss', save_best_only=True)
#early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=1000,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[checkpoint],
    use_multiprocessing=True
)


model_save_path = f'C:/Users/User/Desktop/kunet/multimodal_final.h5'

model.save(model_save_path)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['val_loss'])
plt.ylabel('Losses')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper right')
plt.subplot(1, 2, 2)
plt.plot(history.history['dice_coef'], color='r')
plt.plot(history.history['val_dice_coef'])
plt.ylabel('dice_coef')
plt.xlabel('Epoch')
plt.tight_layout()
plt.savefig(f'C:/Users/User/Desktop/kunet/multimodal_process.png')
plt.close()

max_dice_coef = max(history.history['dice_coef'])
max_val_dice_coef = max(history.history['val_dice_coef'])

f = open(f"C:/Users/User/Desktop/kunet/multimodal_output.txt", "a")
print("max dice coef: ", max_dice_coef, file=f)
print("max val dice coef: ", max_val_dice_coef, file=f)
f.close()