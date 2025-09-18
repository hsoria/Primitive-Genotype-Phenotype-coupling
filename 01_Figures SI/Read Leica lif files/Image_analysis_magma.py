import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.colors import Normalize
from skimage.restoration import rolling_ball 
from skimage.io import imread, imshow
from readlif.reader import LifFile

from PIL import Image, ImageDraw, ImageFont
import os

colors = [  "#56B4E9", "#009E73", "#CC79A7", "#999999", "#E69F00",
            "#DB2B39", "#0076A1", "#0072B2", "#1A5042","#0C1713"]
palette = sns.color_palette(colors)

sns.set_theme(context='notebook', style='ticks', font='Roboto-Light', 
              font_scale=1.3, 
              rc={"lines.linewidth": 1.6, 'axes.linewidth':1.6, 
                  "xtick.major.width":1.6,"ytick.major.width":1.6}, 
              palette=palette)
sns.color_palette(colors)

colors = [  "#56B4E9", "#009E73", "#CC79A7", "#999999", "#E69F00","#DB2B39", "#0076A1", "#0072B2", "#1A5042","#0C1713"]
palette = sns.color_palette(colors)

sns.set_theme(context='notebook', style='ticks', font='Roboto-Light', 
              font_scale=1.3, 
              rc={"lines.linewidth": 1.6, 'axes.linewidth':1.6, 
                                  "xtick.major.width":1.6,"ytick.major.width":1.6}, 
              palette = palette)
sns.color_palette(colors)


def process_multiple_file_kinetics_merge_mean(leica_file, t_focus, rolling_radius):

    output_name = leica_file.replace(".lif", "")
    file = LifFile(leica_file)
    
    # Initialize variables to store data across all files
    files = []
    frames = []
    stacks = []

    # Iterate through each file and accumulate images and frames
    for i in range(file.num_images):
        f1 = file.get_image(i)
        n_frames = f1.nt  # Number of frames for the current image
        
        files.append(f1)
        frames.append(n_frames)

        # Process the z-stacks for each frame in the current file
        for t in range(n_frames):
            z_stacks = [z for z in f1.get_iter_z(c=0, t=t)]
            projection = np.mean(z_stacks, axis=0)  # Average intensity projection
            background_rolling = rolling_ball(projection, radius=rolling_radius)
            projection2 = projection - background_rolling
            stacks.append(projection2)  # Accumulate stacks across files

    # Image processing and time counter
    images = []
    cmap = plt.cm.magma
    scale_bar_length_micrometers = 20.0
    pixel_to_micrometer = files[0].scale[0]  # Assuming the scale is consistent across files
    scale_bar_length_pixels = int(scale_bar_length_micrometers * pixel_to_micrometer)

    total_frames = len(stacks)
    cycle_time = float(files[0].settings["CycleTime"])  # Assuming CycleTime is the same for all files

    for frame_idx, projection in enumerate(stacks):

        # Calculate the elapsed time for the current frame
        elapsed_time_seconds = (frame_idx * cycle_time) + t_focus
        hours = int(elapsed_time_seconds // 3600)
        minutes = int((elapsed_time_seconds % 3600) // 60)
        seconds = int(elapsed_time_seconds % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Create an image with the desired colormap
        norm = Normalize(vmin=projection.min(), vmax=projection.max())
        img = Image.fromarray((cmap(norm(projection)) * 255).astype(np.uint8))

        # Create a drawing context to add the time counter and scale bar to the image
        draw = ImageDraw.Draw(img)
        font_path = "Roboto-Light.ttf"
        font_size = 80
        font = ImageFont.truetype(font_path, font_size)

        # Position and style for the time counter text
        text_x, text_y = 20, 20
        text_color = "white"

        # Position and style for the scale bar
        scale_bar_x, scale_bar_y = 10, img.height - 40
        scale_bar_color = "white"

        # Add the time counter text and scale bar to the image
        draw.text((text_x, text_y), time_str, fill=text_color, font=font)
        draw.rectangle([(scale_bar_x, scale_bar_y), 
                        (scale_bar_x + scale_bar_length_pixels, scale_bar_y + 15)], fill=scale_bar_color)

        images.append(img)  # Append the processed image

    # Specify the output file name
    output_file = f"{output_name}_magma.gif"
    images[0].save(output_file, save_all=True, append_images=images, duration=100, loop=0)

    # Create a time column for plotting
    t_column = np.linspace(t_focus, (total_frames * cycle_time), total_frames)

    return stacks, t_column, images

def process_multiple_file_kinetics_merge_max(leica_file, t_focus, n_samples, rolling_radius):

    """
    Input variables:
    leica_file - name of your lif file
    t_focus - seconds to focus your sample and start measuring
    """
    # Read data from file and metadata
        
    output_name = leica_file.replace(".lif", "")
    file = LifFile(leica_file)
    files = []
    frames = []
    for i in range(n_samples):
        f1 = file.get_image(i)
        n_frames = f1.nt

        files.append(f1)
        frames.append(n_frames)

    stacks = []
    for f, frame_number in zip(files, frames):
        for t in range(frame_number):
            z_stacks = [i for i in f.get_iter_z(c=0, t=t)]
            projection = np.max(z_stacks, axis=0)  # z-stack by average intensity
            background_rolling = rolling_ball(projection, radius=rolling_radius)
            projection2 = projection - background_rolling
            stacks.append(projection2)


    # Create a list to hold image objects
    images = []

    # Define the colormap and normalization settings
    cmap = plt.cm.viridis

    # Define the desired scale bar length in micrometers (10 um)
    scale_bar_length_micrometers = 20.0

    # Calculate the corresponding scale bar length in pixels based on the pixel-to-micrometer ratio
    pixel_to_micrometer = f1.scale[0]
    scale_bar_length_pixels = int(scale_bar_length_micrometers * pixel_to_micrometer)

    # Loop through each numpy array in the "stacks" list
    for frame_idx, projection in enumerate(stacks):

        # Calculate the total time for the experiment
        time_experiment = t_focus + (len(stacks) * float(f1.settings["CycleTime"]))
        time_increment = float(f1.settings["CycleTime"])
        # Calculate the elapsed time for the current frame
        elapsed_time_seconds = (frame_idx * time_increment) + t_focus
        hours = int(elapsed_time_seconds // 3600)
        minutes = int((elapsed_time_seconds % 3600) // 60)
        seconds = int(elapsed_time_seconds % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"


        # Create an image with the desired colormap
        norm = Normalize(vmin=projection.min(), vmax=projection.max())
        img = Image.fromarray((cmap(norm(projection)) * 255).astype(np.uint8))

        # Create a drawing context to add the time counter and scale bar to the image
        draw = ImageDraw.Draw(img)
        # Load font
        font_path = " Roboto-Light.ttf"
        font_size = 60
        font = ImageFont.truetype(font_path, font_size)


        # Position and style for the time counter text
        text_x, text_y = 20, 20
        text_color = "white"



        # Position and style for the scale bar
        scale_bar_x, scale_bar_y = 10, img.height - 40
        scale_bar_color = "white"

        # Add the time counter text using the specified font and size
        draw.text((text_x, text_y), time_str, fill=text_color, font=font)

        # Add the scale bar
        draw.rectangle([(scale_bar_x, scale_bar_y), (scale_bar_x + scale_bar_length_pixels, scale_bar_y + 15)],
                        fill=scale_bar_color)

        images.append(img)

    # Specify the output file name
    output_file = f"{output_name}_magma.gif"
    images[0].save(output_file, save_all=True, append_images=images, duration=100, loop=0, dpi=(600, 600))

    t_column = np.linspace(t_focus, (len(images)*float(f1.settings["CycleTime"])), len(stacks))

    return stacks, t_column, images

def create_mosaic(PIL_images_list, filename):
    images = PIL_images_list
    # Define grid size 
    grid_size = int(np.ceil(np.sqrt(len(images))))  # This calculates a square grid
    image_width, image_height = images[0].size  # Assuming all images have the same size

    # Create a blank canvas to paste images onto
    grid_image = Image.new('RGBA', (grid_size * image_width, grid_size * image_height))

    # Define padding (space between images)
    padding = 0  # You can adjust this value

    # Create a blank canvas to paste images onto, considering padding
    grid_image = Image.new(
        'RGBA', 
        (grid_size * (image_width + padding) - padding, grid_size * (image_height + padding) - padding), 
        (255, 255, 255, 0)  # Optional: background color (white with transparency)
    )
    # Paste each image into the grid
    for idx, img in enumerate(images):
        x = (idx % grid_size) * image_width
        y = (idx // grid_size) * image_height
        grid_image.paste(img, (x, y))

    # Convert PIL image to numpy array to plot with matplotlib
    grid_image_np = np.array(grid_image)

    # Plot the facegrid using matplotlib
    plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(grid_image_np)
    plt.axis('off')  # Hide the axis
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight", transparent = True)
    
def get_one_image(PIL_Image_list, index, file_name, save = False):


    if save == False:

        image = PIL_Image_list[index]
        # Desired size in pixels (width, height)
        new_size = (520, 520)

        # Resize the image
        resized_image = image.resize(new_size, Image.LANCZOS)
        return resized_image
    else:
        image = PIL_Image_list[index]
        # Desired size in pixels (width, height)
        new_size = (2048, 2048)

        # Resize the image
        resized_image = image.resize(new_size, Image.LANCZOS)
        resized_image
        # Save the resized image with 600 DPI
        resized_image.save(f'{file_name}.png', dpi=(600, 600))

