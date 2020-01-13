import os
import cv2
import numpy as np
import click
import matplotlib.pyplot as plt
from gan.model import generator
from gan.utils import preprocess_LR, deprocess_HR

# Remove warnings and debug info
np.warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_generator(weights_path):
    generator = generator_model()
    generator.load_weights(weights_path)
    return generator

def generate_batch(image_LR,model_shape):
    batch_lr = np.zeros((1,model_shape,model_shape,3), dtype=np.float32)
    batch_lr[0] = image_LR
    batch_lr = preprocess_LR(batch_lr)
    #batch_lr = np.transpose(batch_lr, (0,3,1,2))
    return batch_lr

def get_plots(image_name,image_ext,path_train,path_test,path_weights,model_shape):
    # Load LR and HR images
    image_LR = cv2.imread(f'{path_train}{image_name}.{image_ext}')
    image_HR = cv2.imread(f'{path_test}{image_name}.{image_ext}')

    # Load generator and predict result from the LR image
    generator = load_generator(path_weights)
    generated_images = generator.predict(x=generate_batch(image_LR,model_shape))
    #generated_images = np.transpose(generated_images, ((0,2,3,1)))

    # Deprocess images
    image_LR = cv2.cvtColor(image_LR, cv2.COLOR_BGR2RGB).astype(np.uint8)   
    image_generated = cv2.cvtColor(deprocess_HR(generated_images[0]), cv2.COLOR_BGR2RGB).astype(np.uint8)
    image_HR = cv2.cvtColor(image_HR, cv2.COLOR_BGR2RGB).astype(np.uint8)
     
     # Plot LR image, Generated image, HR image for comparison
    fig, axes = plt.subplots(1,3, figsize=(15,7))
    axes[0].imshow(image_LR)
    axes[1].imshow(image_generated)
    axes[2].imshow(image_HR)
    plt.show()

@click.command()
@click.option('--image_name', default='24_5_10', help='Image name to enchance.')
@click.option('--image_ext', default='png', help='Extension of train and test image.')
@click.option('--path_train', default='D:/Project/Thesis/MySRGAN/dataset/train_16_50/', help='Path to training images folder.')
@click.option('--path_test', default='D:/Project/Thesis/MySRGAN/dataset/test_16_50/', help='Path to test images folder.')
@click.option('--path_weights', default='D:/Project/itc-final-thesis-gan/data/weights/generator_16_50_199999.h5', help='Path to generator weights.')
@click.option('--model_shape', default=64, help='Generator model shape, than that would be used as input to the generator.')
def get_examples_for_comparison(image_name,image_ext,path_train,path_test,path_weights,model_shape):
    return get_plots(image_name,image_ext,path_train,path_test,path_weights,model_shape)

if __name__ == "__main__":
    get_examples_for_comparison()
