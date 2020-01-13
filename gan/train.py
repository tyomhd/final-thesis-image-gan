import os
import datetime
import click
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import xlrd
import xlwt 
from xlwt import Workbook 
from time import time
from model import generator as gen, discriminator as disc, gan
from batch_generator import BatchGenerator

def train_multiple_outputs(data_format, 
                        batch_size, 
                        dataset_shuffle, 
                        start_iteration,
                        num_iterations,
                        model_checkpoint_freq_weights,
                        model_checkpoint_freq_loss, 
                        path_hr, 
                        path_lr, 
                        path_weights, 
                        path_discriminator_weights,
                        path_generator_weights, 
                        path_logs_excel, 
                        training_name):

    # Variables to store the loss values
    list_d_loss_fake = []
    list_d_loss_real = []
    list_g_loss_mse_vgg = []
    list_g_loss_cxent = []

    # Create or open the Excel file
    excel_file_name = training_name
    excel_path = (f'{path_logs_excel}loss-{excel_file_name}.xls') 
    excel_counter = 0

    if (os.path.isfile(excel_path)):
        excel_workbook = xlrd.open_workbook(f'{path_logs_excel}loss-{excel_file_name}.xls') 
        excel_sheet = excel_workbook.sheet_by_index(0)
    else:
        excel_workbook = Workbook()
        excel_sheet = excel_workbook.add_sheet(training_name)
        excel_sheet.write(0, 0, 'epoch')
        excel_sheet.write(0, 1, 'd_loss_fake')
        excel_sheet.write(0, 2, 'd_loss_real')
        excel_sheet.write(0, 3, 'g_loss_mse_vgg')
        excel_sheet.write(0, 4, 'g_loss_cxent')

    # Create batch generator
    print('### Status: Creating a Bath Generator...')
    batch_gen = BatchGenerator(
        path_lr=path_lr,
        path_hr=path_hr,
        input_size=64,
        target_size=128,
        data_format=data_format,
        batch_size = batch_size,
        shuffle=dataset_shuffle)

    # Create network model
    print('### Status: Loading GAN model...')
    generator = gen()
    discriminator = disc()

    # Load weights, if needed
    if path_generator_weights:
        print('### Status: Loading Generator weights...')
        generator.load_weights(path_generator_weights)
    if path_discriminator_weights:
        print('### Status: Loading Discriminator weights...')
        discriminator.load_weights(path_discriminator_weights)
        
    network_model = gan(generator, discriminator)

    print('### Status: Starting training...')
    for epoch in tqdm.tqdm(range(num_iterations)):
        ##### Discriminator training #####
        discriminator.trainable = True

        # Generate a batch
        batch_LR, batch_HR = batch_gen.next()
        
        batch_SR = generator.predict(batch_LR)
        labels_SR = np.random.uniform(0.0, 0.25, size=batch_SR.shape[0]).astype(np.float32)
        labels_HR = np.random.uniform(0.75, 1.0, size=batch_HR.shape[0]).astype(np.float32)

        # Train the Discriminator model
        d_loss_fake = discriminator.train_on_batch(batch_SR, labels_SR)
        d_loss_real = discriminator.train_on_batch(batch_HR, labels_HR)

        ##### Generator training #####
        discriminator.trainable = False

        # Generate a batch
        batch_LR, batch_HR = batch_gen.next()

        # Create a labels for HR batch
        labels_HR_G = np.ones((batch_LR.shape[0], 1), dtype = np.float32)
        # Train the Generator model
        g_loss = network_model.train_on_batch(batch_LR, [batch_HR, labels_HR_G])

        # Save model losses
        list_d_loss_fake.append(d_loss_fake)
        list_d_loss_real.append(d_loss_real)
        list_g_loss_mse_vgg.append(g_loss[0])
        list_g_loss_cxent.append(g_loss[1])

        ##### Save weights #####
        if (epoch % model_checkpoint_freq_weights == 0 and epoch != start_iteration) or epoch == num_iterations - 1:
            generator.save_weights(f'{path_weights}generator_{training_name}_{epoch+start_iteration}.h5', True)
            discriminator.save_weights(f'{path_weights}discriminator_{training_name}_{epoch+start_iteration}.h5', True)

        ##### Save logs to Excel file #####
        if epoch != 0 and (epoch + start_iteration) % model_checkpoint_freq_loss == 0:
            # Calculate mewan losses
            d_loss_fake = np.mean(list_d_loss_fake[-model_checkpoint_freq_loss::]).item()
            d_loss_real =np.mean(list_d_loss_real[-model_checkpoint_freq_loss::]).item()
            g_loss_mse_vgg = np.mean(list_g_loss_mse_vgg[-model_checkpoint_freq_loss::]).item()
            g_loss_cxent = np.mean(list_g_loss_cxent[-model_checkpoint_freq_loss::]).item()
            # Write to excel file
            excel_counter += 1
            excel_sheet.write(excel_counter, 0, epoch + start_iteration)
            excel_sheet.write(excel_counter, 1, d_loss_fake)
            excel_sheet.write(excel_counter, 2, d_loss_real)
            excel_sheet.write(excel_counter, 3, g_loss_mse_vgg)
            excel_sheet.write(excel_counter, 4, g_loss_cxent)
            excel_workbook.save(f'loss-{excel_file_name}.xls') 

@click.command()
@click.option('--data_format', default='channels_last', help='The image data format.')
@click.option('--batch_size', default=24, help='A batch size.')
@click.option('--dataset_shuffle', default=1, help='A flag to shuffle the dataset.')
@click.option('--start_iteration', default=0, help='A strarting iteration.')
@click.option('--num_iterations', default=1000000, help='A number of training iterations.')
@click.option('--model_checkpoint_freq_weights', default=1000, help='A frequency of iterations to save the model weigts.')
@click.option('--model_checkpoint_freq_loss', default=50, help='A frequency of iterations to save the model loss.')
@click.option('--path_hr', default='D:/Project/Thesis/MySRGAN/dataset/test_16_50/', help='A path to high-resolution samples.')
@click.option('--path_lr', default='D:/Project/Thesis/MySRGAN/dataset/train_16_50/', help='A path to low-resolution samples.')
@click.option('--path_weights', default='D:/Project/itc-final-thesis-gan/data/weights/', help='A path to weights folder.')
@click.option('--path_discriminator_weights', default='', help='A path to Discriminator model weights.')
@click.option('--path_generator_weights', default='', help='A path to Generator model weights.')
@click.option('--path_logs_excel', default='D:/Project/itc-final-thesis-gan/notes/', help='A path to save excel file with loss logs.')
@click.option('--training_name', default='16_50_batch24', help='A name of the current training.')
def train_command(data_format, 
                batch_size, 
                dataset_shuffle, 
                start_iteration, 
                num_iterations, 
                model_checkpoint_freq_weights,
                model_checkpoint_freq_loss, 
                path_hr, 
                path_lr, 
                path_weights, 
                path_discriminator_weights,
                path_generator_weights, 
                path_logs_excel, 
                training_name):

    return train_multiple_outputs(data_format, 
                                batch_size, 
                                bool(dataset_shuffle), 
                                start_iteration,
                                num_iterations - start_iteration,
                                model_checkpoint_freq_weights,
                                model_checkpoint_freq_loss, 
                                path_hr, 
                                path_lr, 
                                path_weights, 
                                path_discriminator_weights,
                                path_generator_weights, 
                                path_logs_excel, 
                                training_name)

if __name__ == '__main__':
    train_command()
