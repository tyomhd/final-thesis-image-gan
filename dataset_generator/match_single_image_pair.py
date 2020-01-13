import os
import click
from PIL import Image
from PIL.ExifTags import TAGS
from image_processor import process_matching_images

@click.command()
@click.option('--hr_img_path', default='D:/Project/itc-final-thesis-gan/data_final/pictures-final-all/DSC04640.JPG', help='A path with high-resolution images.')
@click.option('--lr_img_path', default='D:/Project/itc-final-thesis-gan/data_final/pictures-final-all/DSC04647.JPG', help='A path with low-resolution images.')
@click.option('--path_to_save', default='D:/Project/itc-final-thesis-gan/data/thesis_examples/', help='A path to save the example.')
@click.option('--fast_matching', default=1, help='todo')
def process_image_pair(hr_img_path, lr_img_path, path_to_save, fast_matching):
    print('Processing...')
    fast_matching = bool(fast_matching)
    print(f"Fast matching enabled: {fast_matching}")
    process_matching_images(hr_img_path, lr_img_path, path_to_save, fast_matching, process_as_grid=False, index=0)
    print('Done')

if __name__ == "__main__":
    process_image_pair()