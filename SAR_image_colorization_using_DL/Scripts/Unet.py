# CELL 1: Loading Data
import numpy as np

# Load the numpy arrays
sar_images = np.load('preprocessed_sar_images_1K.npy')
optical_images = np.load('preprocessed_optical_images_1K.npy')

# Check the shape of the loaded data
print(sar_images.shape)
print(optical_images.shape)

# CELL 2: U-Net Model Definition
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def conv_block(inputs, num_filters):
    """Convolutional block consisting of two Conv2D layers followed by Batch Normalization and ReLU activation."""
    x = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def encoder_block(inputs, num_filters):
    """Encoder block with max pooling."""
    x = conv_block(inputs, num_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """Decoder block with Conv2DTranspose and skip connections."""
    x = Conv2DTranspose(num_filters, kernel_size=2, strides=2, padding='same')(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(3, kernel_size=1, padding='same', activation='tanh')(d4)

    model = Model(inputs, outputs, name='U-Net')
    return model

# Initialize U-Net model
input_shape = (256, 256, 1)  # Shape of SAR images
unet_model = build_unet(input_shape)
unet_model.summary()

# CELL 3: Preprocessing Data
def preprocess_data(sar_images, optical_images):
    sar_images = sar_images.astype(np.float32) / 127.5 - 1.0
    optical_images = optical_images.astype(np.float32) / 127.5 - 1.0
    return sar_images, optical_images

sar_images, optical_images = preprocess_data(sar_images, optical_images)

dataset = tf.data.Dataset.from_tensor_slices((sar_images, optical_images))
dataset = dataset.shuffle(buffer_size=len(sar_images)).batch(8)  # Adjust the batch size if needed

# CELL 4: Loss Function and Optimizer
def unet_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

optimizer = Adam(2e-4, beta_1=0.5)

unet_model.compile(optimizer=optimizer, loss=unet_loss)

# CELL 5: Training the Model
epochs = 20
history = unet_model.fit(dataset, epochs=epochs)

# Save the trained U-Net model
unet_model.save('unet_colorization_model.keras')

# CELL 6: Evaluation
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_model(model, sar_images, optical_images):
    psnr_values = []
    ssim_values = []

    for i in range(len(sar_images)):
        generated_image = model.predict(sar_images[i:i+1])
        target_image = optical_images[i]

        # Rescale images to 0-1 for metric calculations
        generated_image = (generated_image + 1) / 2.0
        target_image = (target_image + 1) / 2.0

        # Ensure the images are the same size
        if target_image.shape != generated_image.shape:
            print(f"Skipping image {i}: shape mismatch {target_image.shape} vs {generated_image.shape}")
            continue

        # Calculate PSNR and SSIM
        psnr_value = psnr(target_image, generated_image, data_range=generated_image.max() - generated_image.min())
        ssim_value = ssim(target_image, generated_image, multichannel=True)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    print(f'Average PSNR: {np.mean(psnr_values)}')
    print(f'Average SSIM: {np.mean(ssim_values)}')

evaluate_model(unet_model, sar_images, optical_images)

# CELL 7: Visualization
import matplotlib.pyplot as plt

def plot_samples(model, sar_images, optical_images, num_samples=5):
    plt.figure(figsize=(15, 5))

    for i in range(num_samples):
        idx = np.random.randint(0, len(sar_images))
        sar_image = sar_images[idx]
        optical_image = optical_images[idx]
        generated_image = model.predict(sar_image[np.newaxis, ...])[0]

        # Rescale images to 0-1 for visualization
        sar_image = (sar_image + 1) / 2.0
        optical_image = (optical_image + 1) / 2.0
        generated_image = (generated_image + 1) / 2.0

        plt.subplot(3, num_samples, i+1)
        plt.imshow(sar_image.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title("SAR Image")

        plt.subplot(3, num_samples, num_samples + i + 1)
        plt.imshow(generated_image)
        plt.axis('off')
        plt.title("Generated Image")

        plt.subplot(3, num_samples, 2 * num_samples + i + 1)
        plt.imshow(optical_image)
        plt.axis('off')
        plt.title("Ground Truth")

    plt.tight_layout()
    plt.show()

plot_samples(unet_model, sar_images, optical_images)
