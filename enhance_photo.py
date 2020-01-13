import os
import math
import numpy as np
import matplotlib.pyplot as plt
import click
import tqdm
from PIL import Image
from gan.model import generator
from gan.utils import preprocess_LR, deprocess_HR

# Remove warnings and debug info
np.warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate_batch(image_path,model_shape,batch_size):
    print('### Status: Preprocessing the image')
    input_image = plt.imread(image_path)
    height, width, color = input_image.shape
    y_grid = math.floor(height/model_shape)
    x_grid = math.floor(width/model_shape)
    y_shift = math.floor(height%model_shape/2)
    x_shift = math.floor(width%model_shape/2)
    num_samples = math.floor(x_grid*y_grid)
    batches_per_column = math.floor(y_grid/batch_size)
    column_tail = y_grid - (batches_per_column * batch_size)
    print(f'Height: {height} px, Loss: {height%model_shape} px')
    print(f'Width: {width} px, Loss: {width%model_shape} px')
    print(f'Grid size : {y_grid}x{x_grid}')
    print(f'Tail : {column_tail}')

    samples_batch = np.zeros((num_samples,model_shape,model_shape,3), dtype=np.float32)
    for x in range(x_grid):
        for y in range(y_grid):
            num = x*y_grid + y
            y_start = y*model_shape + y_shift
            x_start = x*model_shape + x_shift
            samples_batch[num] = input_image[y_start:y_start+model_shape,x_start:x_start+model_shape,:]
    
    samples_batch = preprocess_LR(samples_batch)

    return samples_batch, num_samples, batches_per_column, column_tail

def load_generator(weights_path):
    print('### Status: Loading Generator')
    g = generator()
    g.load_weights(weights_path)
    return g

def _list_valid_filenames_in_path(path):
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

    filenames = []
    basedir = os.path.dirname(path)
    for root, _, files in _recursive_list(path):
        for fname in files:
            absolute_path = os.path.join(root, fname)
            filenames.append(os.path.relpath(absolute_path, basedir))
    return filenames

def enchance_image(image_path, weights_path, model_shape, target_shape, batch_size):
    image_names_list = _list_valid_filenames_in_path(image_path)
    
    g = load_generator(weights_path)
    
    for img in image_names_list:
        path, ext = img.split('.')
        samples_batch, num_samples, batches_per_column, column_tail = generate_batch(image_path+img,model_shape,batch_size)
        num_batches = math.ceil(num_samples/batch_size)
        pericted_samples = [None]*num_samples
        print(f'### Status: Predicting image: {img}')
        for x in tqdm.tqdm(range(num_batches)):
            batch_start = x * batch_size
            batch_stop = batch_start + batch_size if batch_start + batch_size < num_samples else num_samples
            pericted_samples[batch_start:batch_stop] = g.predict(x=samples_batch[batch_start:batch_stop])
        print('### Status: Deprocessing image')
        samples_per_column = batches_per_column * batch_size + column_tail
        num_iterations = math.floor(num_samples/samples_per_column)
        predicted_columns = [None]*num_iterations
        for x in tqdm.tqdm(range(num_iterations)):
            batch_start = x * samples_per_column
            batch_stop = batch_start + samples_per_column 

            current_column = pericted_samples[batch_start:batch_stop]
            predicted_column = np.vstack((np.asarray(deprocess_HR(i)) for i in current_column))
            predicted_columns[x] = predicted_column

        output = np.hstack(predicted_columns)
        output = output.astype(np.uint8)
        Image.fromarray(output).save(f'{image_path}{img}_echanced.jpg', quality=100)

    print('### Status: Done')

@click.command()
@click.option('--input_folder', default='C:/Users/Admin/Desktop/SR/', help='Folder with input images to enhance.')
@click.option('--weights_path', default='D:/Project/itc-final-thesis-gan/data_final/training/final-16-50-batch24/weights/generator_16_50_batch24_final.h5', help='Path to the generator weights.')
@click.option('--model_shape', default=64, help='Generator model shape, than that would be used as input to the generator.')
@click.option('--target_shape', default=128, help='A target shape size.')
@click.option('--batch_size', default=24, help='A batch size would be used during the generator prediction.')
def enchance_command(input_folder, weights_path, model_shape, target_shape, batch_size):
    enchance_image(input_folder, weights_path, model_shape, target_shape, batch_size)

if __name__ == '__main__':
    enchance_command()
