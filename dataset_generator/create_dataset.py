import os
import click
from PIL import Image
from PIL.ExifTags import TAGS
from image_processor import process_matching_images

def reorganize_files(image_format, focal_length_lr, focal_length_hr, path_input, path_output):
    images_lr = list()
    images_hr = list()

    # Scan input image location and sort images to two categories by image focal length
    for f in sorted(os.listdir(path_input + '.')):
        if f.endswith(image_format):
            img = Image.open(path_input + f)
            focal_lenght = get_exif(img,'FocalLength')
            focal_lenght = int(focal_lenght[0] / focal_lenght [1])
            if focal_lenght > focal_length_lr-3 and focal_lenght < focal_length_lr+3:
                images_lr.append(f)
            elif focal_lenght > focal_length_hr-3 and focal_lenght < focal_length_hr+3:
                images_hr.append(f)

    # If number of images is the same in both categories process images to create dataset
    if(len(images_lr) == len(images_hr)):
        num_samples = 0
        for i in range(len(images_lr)):
            lr_img_path = path_input + images_lr[i]
            hr_img_path = path_input + images_hr[i]

            print(f'Processing images: {images_hr[i]} and {images_lr[i]}')
            num_samples += process_matching_images(hr_img_path, lr_img_path, path_output, True, True, i, focal_length_lr, focal_length_hr)
        
        print('Dataset is ready')
        print(f'Total amount of paired images: {len(images_hr)}')
        print(f'Total amount of samples in each category: {num_samples}')
    else:
        print(f'Unequal amount of paired images. {focal_length_lr}mm: {len(images_lr)}, {focal_length_hr}mm: {len(images_hr)}')


def get_exif (img,field) :
    exif = img._getexif()
    for (k,v) in exif.items():
        if TAGS.get(k) == field:
            return v

@click.command()
@click.option('--image_format', default='.JPG', help='An image format for the dataset.')
@click.option('--focal_length_lr', default=16, help='A focal length for low-resolution samples.')
@click.option('--focal_length_hr', default=50, help='A focal length for high-resolution samples.')
@click.option('--path_input', default='D:/Project/itc-final-thesis-gan/data_final/pictures-final-used-in-final-training/', help='A path with images folder to create the dataset from.')
@click.option('--path_output', default='D:/Project/itc-final-thesis-gan/dataset/', help='An output location for saving the dataset.')
def create_dataset(image_format, focal_length_lr, focal_length_hr, path_input, path_output):
    print('Starting...')
    print(f'Working with focal length {focal_length_lr}-{focal_length_hr}.')
    reorganize_files(image_format, focal_length_lr, focal_length_hr, path_input, path_output)


if __name__ == "__main__":
    create_dataset()
