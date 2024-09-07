import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, LeakyReLU, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Load Data
def load_data(sar_path, optical_path):
    sar_images = np.load(sar_path)
    optical_images = np.load(optical_path)
    return sar_images, optical_images

# Preprocess Data
def preprocess_data(sar_images, optical_images):
    sar_images = sar_images.astype(np.float32) / 127.5 - 1.0
    optical_images = optical_images.astype(np.float32) / 127.5 - 1.0
    return sar_images, optical_images

# Build Generator
def build_generator():
    inputs = Input(shape=(256, 256, 1))  # SAR images are single-channel

    # Encoder
    x = Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Decoder
    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(3, 4, strides=2, padding='same')(x)
    outputs = tf.keras.layers.Activation('tanh')(x)

    return Model(inputs, outputs)

# Build Discriminator
def build_discriminator():
    inputs = Input(shape=[256, 256, 3])
    
    # Downsampling layers
    x = Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 4, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(1, 4, strides=1, padding='same')(x)
    outputs = tf.keras.layers.Activation('sigmoid')(x)

    return Model(inputs, outputs)

# Loss Functions
def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

# Initialize Models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
generator_optimizer = Adam(2e-4, beta_1=0.5)
discriminator_optimizer = Adam(2e-4, beta_1=0.5)

# Dataset Preparation
def prepare_dataset(sar_images, optical_images):
    sar_images, optical_images = preprocess_data(sar_images, optical_images)
    dataset = tf.data.Dataset.from_tensor_slices((sar_images, optical_images))
    dataset = dataset.shuffle(buffer_size=100).batch(1)
    return dataset

# Training Step
@tf.function
def train_step(input_image, target_image):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(input_image, training=True)

        real_output = discriminator(target_image, training=True)
        fake_output = discriminator(generated_image, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Training Loop
def train_model(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            input_image, target_image = image_batch
            gen_loss, disc_loss = train_step(input_image, target_image)
        print(f'Epoch {epoch+1}/{epochs} - Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}')

    # Save models
    generator.save('generator_pix2pix_model.keras')
    discriminator.save('discriminator_pix2pix_model.keras')

# Evaluation
def evaluate_model(generator, sar_images, optical_images):
    psnr_values = []
    ssim_values = []

    for i in range(len(sar_images)):
        generated_image = generator.predict(sar_images[i:i+1])
        target_image = optical_images[i]

        generated_image = (generated_image + 1) / 2.0
        target_image = (target_image + 1) / 2.0

        if generated_image.shape[-1] == 1:
            generated_image = np.repeat(generated_image, 3, axis=-1)

        if target_image.shape != generated_image.shape:
            continue

        psnr_value = psnr(target_image, generated_image, data_range=generated_image.max() - generated_image.min())
        ssim_value = ssim(target_image, generated_image, multichannel=True)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f'Average PSNR: {avg_psnr}')
    print(f'Average SSIM: {avg_ssim}')

# Plot Samples
def plot_samples(generator, sar_images, optical_images, num_samples=5):
    plt.figure(figsize=(15, 5))

    for i in range(num_samples):
        idx = np.random.randint(0, len(sar_images))
        sar_image = sar_images[idx]
        optical_image = optical_images[idx]
        generated_image = generator.predict(sar_image[np.newaxis, ...])[0]

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

# Main Function
def main():
    sar_path = 'path_to_sar_images.npy'
    optical_path = 'path_to_optical_images.npy'
    
    sar_images, optical_images = load_data(sar_path, optical_path)
    dataset = prepare_dataset(sar_images, optical_images)

    train_model(dataset, epochs=20)

    # Load the saved model
    generator = tf.keras.models.load_model('generator_pix2pix_model.keras')
    
    # Evaluate the model
    evaluate_model(generator, sar_images, optical_images)

    # Plot samples
    plot_samples(generator, sar_images, optical_images)

if __name__ == '__main__':
    main()
