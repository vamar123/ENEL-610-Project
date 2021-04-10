import face_recognition
import os
import cv2
import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''

*** This version is to plot distribution histogram for 1:N match ***

IMPORTANT NOTE:
- DO NOT use 'cnn' model if you don't have a powerful GPU. Otherwise it will be extremely slow!
- DO NOT include multiple faces in one picture. Otherwise the export will be inexecutable.
- The pictures in 'known' folder should be ONE PERSON and be placed in ONE SUBFOLDER.
- The pictures in 'unknown' folder should comprise of EQUAL NUMBERS of photos for different persons.
- Remember to change the TRIAL_PER_PERSON and PERSONS_FOR_TRIAL parameter to the correct value for reshape the array.
- Remember to change the NAME_OF_ORIGIN and NAME_OF_TRIALS parameter to the correct value for plot and export.
- The sequence of NAME_OF_TRIALS should be sorted as the corresponding file name (sorted alphabetically).
- There should be one and only one face in one picture. Otherwise an error will occur.
'''

# Program parameters initialization
KNOWN_FACES_DIR = 'known'
UNKNOWN_FACES_DIR = 'unknown'
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
PERSONS_FOR_TRIAL = 8  # For reshaping the face_distance array. BE SURE TO VERIFY IT BEFORE RUNNING!
TRIAL_PER_PERSON = 3  # For reshaping the face_distance array. BE SURE TO VERIFY IT BEFORE RUNNING!
NAME_OF_ORIGIN = "Angela"  # For plot title. BE SURE TO VERIFY IT BEFORE RUNNING!
NAME_OF_TRIALS = ["Aaron", "Obama", "Chris", "Eddie", "Laurence", "Shawn", "Scarlett", "Angela"]  # For plot legend. BE SURE TO VERIFY IT BEFORE RUNNING!

# Variables initialization
known_faces = []
known_names = []
known_faces_count = 0
unknown_faces_count = 0
known_filenames = []
unknown_filenames = []
total_distance = []

# We organize known faces as sub-folders of KNOWN_FACES_DIR
# Each sub-folder's name becomes our label (name)
print('Loading known faces...')
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        known_filenames.append(f'{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)
        known_faces_count += 1

print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
for filename in os.listdir(UNKNOWN_FACES_DIR):

    # Load image
    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
    unknown_filenames.append(f'{filename}')

    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)

    # Now since we know locations, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)

    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # But this time we assume that there might be more faces in an image - we can find faces of different people
    print(f', found {len(encodings)} face(s)')
    print('Matching Result:')
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')
        else:
            print('Unknown face detected.')

        print('Distance data:')
        print(face_recognition.face_distance(known_faces, face_encoding))
        current_distance = np.array(face_recognition.face_distance(known_faces, face_encoding))
        total_distance = np.r_[total_distance, current_distance]
        unknown_faces_count += 1

total_distance = np.reshape(total_distance, (unknown_faces_count, known_faces_count))
total_distance_plot = np.reshape(total_distance, (PERSONS_FOR_TRIAL, TRIAL_PER_PERSON * known_faces_count))

df_distance = pd.DataFrame(total_distance)
df_distance_plot = pd.DataFrame(total_distance_plot.T)

# Add index and column name
df_distance.columns = [known_filenames]
df_distance.index = [unknown_filenames]
df_distance_plot.columns = NAME_OF_TRIALS

# Create and writer pd.DataFrame to excel
print('Exporting face_distance data to Distance_Array.xlsx ...')
writer = pd.ExcelWriter('Distance_Array.xlsx')
df_distance.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()
print('Exporting face_distance data to Distance_Array_Plot.xlsx ...')
writer = pd.ExcelWriter('Distance_Array_Plot.xlsx')
df_distance_plot.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()

# Plot histogram and fitting curve
plot_df = pd.read_excel('Distance_Array_Plot.xlsx', index_col=0)
sb.displot(plot_df, kind="kde", cut=0)
plt.suptitle("Distribution Curve of Matching with " + NAME_OF_ORIGIN)
sb.displot(plot_df, bins=15, alpha=0.5)
plt.suptitle("Histogram of Matching with " + NAME_OF_ORIGIN)
plt.show()
