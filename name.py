import os

# Set the directory path and the prefix for the new file names
directory = 'C:\\Captcha1\\kriat.datoset\\117641_1000'
prefix = '117641_variant'

# Get a list of all files in the directory
files = os.listdir(directory)

# Iterate over the files and rename them
for i, file in enumerate(files, start=1):
    new_name = f'{prefix}{i}.jpg'
    os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
