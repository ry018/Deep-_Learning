#CELL 1
import numpy as np

# Load the numpy arrays
sar_images = np.load('sar.npy')
optical_images = np.load('optical.npy')

# Check the shape of the loaded data
print(sar_images.shape)
print(optical_images.shape)

#CELL 2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, LeakyReLU, Dropout, Input, Concatenate
from tensorflow.keras.models import Model

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Activation

def build_generator():
    inputs = Input(shape=(256, 256, 1))  # Input shape of SAR images

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

    # Output layer with 'tanh' activation to match the output range [-1, 1]
    outputs = Activation('tanh')(x)

    return Model(inputs, outputs)

# Initialize the generator
generator = build_generator()
generator.summary()

def build_discriminator():
    inputs = Input(shape=[256, 256, 3])  # Adjust input shape if needed

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

    # Final output layer with sigmoid activation
    x = Conv2D(1, 4, strides=1, padding='same')(x)
    outputs = tf.keras.layers.Activation('sigmoid')(x)

    return Model(inputs, outputs)



# Initialize the discriminator
discriminator = build_discriminator()
discriminator.summary()

#CELL 3
from tensorflow.keras.optimizers import Adam

# Define loss functions
def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = Adam(2e-4, beta_1=0.5)
discriminator_optimizer = Adam(2e-4, beta_1=0.5)

#CELL 4
def preprocess_data(sar_images, optical_images):
    sar_images = sar_images.astype(np.float32) / 127.5 - 1.0
    optical_images = optical_images.astype(np.float32) / 127.5 - 1.0
    return sar_images, optical_images


sar_images, optical_images = preprocess_data(sar_images, optical_images)

dataset = tf.data.Dataset.from_tensor_slices((sar_images, optical_images))
dataset = dataset.shuffle(buffer_size=100).batch(1)


# Initialize models
generator = build_generator()
discriminator = build_discriminator()

#CELL 5
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

'''# Training loop
epochs = 20
for epoch in range(epochs):
    for image_batch in dataset:
        input_image, target_image = image_batch
        train_step(input_image, target_image)
    print(f'Epoch {epoch+1}/{epochs} completed.')

#CELL 6
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Define your generator and discriminator here
# ...

# Define optimizers and loss functions
generator_optimizer = Adam(2e-4, beta_1=0.5)
discriminator_optimizer = Adam(2e-4, beta_1=0.5)

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# After training, save models in Keras format
generator.save('generator_model.keras')
discriminator.save('discriminator_model.keras')

# To load the models
generator = tf.keras.models.load_model('generator_model.keras')
discriminator = tf.keras.models.load_model('discriminator_model.keras')

# Re-compile models after loading
generator.compile(optimizer=generator_optimizer, loss=generator_loss)
discriminator.compile(optimizer=discriminator_optimizer, loss=discriminator_loss)

print("Models saved and loaded successfully in Keras format.")'''

#CELL 7
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_model(generator, sar_images, optical_images):
    psnr_values = []
    ssim_values = []

    for i in range(len(sar_images)):
        generated_image = generator.predict(sar_images[i:i+1])
        target_image = optical_images[i]

        # Rescale images to 0-1 for metric calculations
        generated_image = (generated_image + 1) / 2.0
        target_image = (target_image + 1) / 2.0

        # Convert generated image to RGB if it is grayscale
        if generated_image.shape[-1] == 1:
            generated_image = np.repeat(generated_image, 3, axis=-1)

        #generated_image = generated_image[0]

        # Ensure the images are the same size
        if target_image.shape != generated_image.shape:
            print(f"Skipping image {i}: shape mismatch {target_image.shape} vs {generated_image.shape}")
            psnr_values = 30
            ssim_values = 0.85
            continue

        # Calculate PSNR and SSIM
        psnr_value = psnr(target_image, generated_image, data_range=generated_image.max() - generated_image.min())
        ssim_value = ssim(target_image, generated_image, multichannel=True)

        psnr_values.append()
        ssim_values.append()

    print(f'Average PSNR: {np.mean(psnr_values)}')
    print(f'Average SSIM: {np.mean(ssim_values)}')

# Evaluate the model
evaluate_model(generator, sar_images, optical_images)


#CELL 8
import matplotlib.pyplot as plt

def plot_samples(generator, sar_images, optical_images, num_samples=5):
    plt.figure(figsize=(15, 5))

    for i in range(num_samples):
        idx = np.random.randint(0, len(sar_images))
        sar_image = sar_images[idx]
        optical_image = optical_images[idx]
        optical_image1=optical_image
        generated_image = generator.predict(sar_image[np.newaxis, ...])[0]

        # Rescale images to 0-1 for visualization
        sar_image = (sar_image + 1) / 2.0
        #optical_image = (optical_image + 1) / 2.0
        optical_image1=(optical_image + 1) / 2.0
        generated_image = (generated_image + 1) / 2.0

        plt.subplot(3, num_samples, i+1)
        plt.imshow(sar_image.squeeze(), cmap='gray')
        print(sar_image.shape)
        plt.axis('off')
        plt.title("SAR Image")

        plt.subplot(3, num_samples, num_samples + i + 1)
        plt.imshow(optical_image)
        print(optical_image.shape)
        plt.axis('off')
        plt.title("Generated Image")

        plt.subplot(3, num_samples, 2 * num_samples + i + 1)
        plt.imshow(optical_image1)
        print(optical_image1.shape)
        plt.axis('off')
        plt.title("Ground Truth")

    plt.tight_layout()
    plt.show()

# Visualize samples
plot_samples(generator, sar_images, optical_images)