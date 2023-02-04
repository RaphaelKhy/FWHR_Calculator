import math

from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import face_recognition
import os
import urllib.request
import pandas as pd


def load_image(path, url=False):
    """Load an image from a file path or a URL.

    Args:
        path (str): The path to the image file or a URL to download the image from.
        url (bool, optional): Indicates whether the `path` is a file path or a URL. Defaults to `False`.

    Returns:
        The loaded image in the form of a numpy array.

    Raises:
        ValueError: If the file type of the image is not .jpg, .png or if the URL is not accessible.
    """
    if not url:
        return face_recognition.load_image_file(path)
    else:
        if path[-3:] == 'jpg' or path[-3:] == 'png':
            urllib.request.urlretrieve(path, 'tmp.jpg')
            return face_recognition.load_image_file('tmp.jpg')
        elif path[-3:] == 'png':
            urllib.request.urlretrieve(path, 'tmp.png')
            return face_recognition.load_image_file('tmp.png')
        else:
            raise ValueError("Unknown image type")


def get_face_points(points, method='average', top='eyebrow'):
    """
    Calculates the coordinates for the corners of the "FWHR" box.

    Args:
        points (list): A list of coordinates for facial landmarks
        method (str, optional): Method to calculate the height of the box. Either 'average' (default), 'left' or 'right'
        top (str, optional): Top line of the box, either based on the bottom of the eyebrows ('eyebrow') or the eyelids ('eyelid', default)

    Returns:
        dict: A dictionary with the top-left, bottom-left, top-right, and bottom-right corner coordinates of the "FWHR" box.

    Note 1:
    It is possible to calculate the top line based on either the bottom of the eyebrows (top = "eyebrow") or the eyelids (top = "eyelid").

    Note 2:
    To counter-act small amounts of rotation it will by default take the average between the height of the two top points.
    """
    width_left, width_right = points[0], points[16]

    if top == 'eyebrow':
        top_left = points[18]
        top_right = points[25]

    elif top == 'eyelid':
        top_left = points[37]
        top_right = points[43]

    else:
        raise ValueError('Invalid top point, use either "eyebrow" or "eyelid"')

    bottom_left, bottom_right = points[50], points[52]

    if method == 'left':
        coords = (width_left[0], width_right[0], top_left[1], bottom_left[1])

    elif method == 'right':
        coords = (width_left[0], width_right[0], top_right[1], bottom_right[1])

    else:
        top_average = int((top_left[1] + top_right[1]) / 2)
        bottom_average = int((bottom_left[1] + bottom_right[1]) / 2)
        coords = (width_left[0], width_right[0], top_average, bottom_average)

    # Move the line just a little above the top of the eye to the eyelid
    if top == 'eyelid':
        coords = (coords[0], coords[1], coords[2] - 4, coords[3])

    return {'top_left': (coords[0], coords[2]),
            'bottom_left': (coords[0], coords[3]),
            'top_right': (coords[1], coords[2]),
            'bottom_right': (coords[1], coords[3])
            }


def good_picture_check(p, debug=False):
    """
    This function checks whether a picture contains a person that is looking straight at the camera.

    Args:
        p (list of tuples): A list of (x, y) tuples representing facial landmarks.
        debug (bool, optional): A flag to enable debug output. Defaults to False.

    Returns:
        bool: A boolean indicating whether the picture contains a person looking straight at the camera.
    """
    # To scale for picture size
    width_im = (p[16][0] - p[0][0]) / 100

    # Difference in height between eyes
    eye_y_l = (p[37][1] + p[41][1]) / 2.0
    eye_y_r = (p[44][1] + p[46][1]) / 2.0
    eye_dif = (eye_y_r - eye_y_l) / width_im

    # Difference top / bottom point nose
    nose_dif = (p[30][0] - p[27][0]) / width_im

    # Space between face-edge to eye, left vs. right
    left_space = p[36][0] - p[0][0]
    right_space = p[16][0] - p[45][0]
    space_ratio = left_space / right_space

    if debug:
        print(eye_dif, nose_dif, space_ratio)

    # These rules are not perfect, determined by trying a bunch of "bad" pictures
    if eye_dif > 5 or nose_dif > 3.5 or space_ratio > 3:
        return False
    else:
        return True


def FWHR_calc(corners):
    """Calculates the facial width-to-height ratio (FWHR) based on the given corners.

    Args:
        corners (dict): A dictionary that contains the coordinates of the four corners of a face in an image. The keys of the dictionary should be 'top_left', 'top_right', 'bottom_left', and 'bottom_right'. The values are the (x, y) coordinates of each corner.

    Returns:
        float: The calculated FWHR value.
    """
    width = corners['top_right'][0] - corners['top_left'][0]
    height = corners['bottom_left'][1] - corners['top_left'][1]
    return float(width) / float(height)


def show_box(image, corners):
    """Draws a box around the facial width height ratio (FWHR) on the input image.

    Args:
        image (numpy array): The input image as a numpy array.
        corners (dict): A dictionary containing the coordinates of the corners of the box.
                        The keys should be 'bottom_left', 'bottom_right', 'top_left', 'top_right'.

    Returns:
        None: The function displays the original image with the box drawn around the FWHR.
    """
    pil_image = Image.fromarray(image)
    w, h = pil_image.size

    # Automatically determine width of the line depending on size of picture
    line_width = math.ceil(h / 100)

    d = ImageDraw.Draw(pil_image)
    d.line([corners['bottom_left'], corners['top_left']], width=line_width)
    d.line([corners['bottom_left'], corners['bottom_right']], width=line_width)
    d.line([corners['top_left'], corners['top_right']], width=line_width)
    d.line([corners['top_right'], corners['bottom_right']], width=line_width)

    imshow(pil_image)


# This function combines all the previous logic into one function.
def get_fwhr(image_path, url=False, show=True, method='average', top='eyelid'):
    """Calculates fwh ratio of a single image

    Args:
        image_path (string): path or URL to image
        url (bool, optional): set to `True` if `image_path` is a url. Defaults to False.
        show (bool, optional): set to `False` if you only want it to return the FWHR. Defaults to True.
        method (str, optional): determines which eye to use for the top point: `left`, `right`, or `average`. Defaults to 'average'.
        top (str, optional): determines whether to use the `eyebrow` as top point or `eyelid` as top point. Defaults to 'eyelid'.

    Returns:
        double: fwh ratio
    """
    image = load_image(image_path, url)
    landmarks = face_recognition.api._raw_face_landmarks(image)
    landmarks_as_tuples = [(p.x, p.y) for p in landmarks[0].parts()]

    if good_picture_check(landmarks_as_tuples):
        corners = get_face_points(landmarks_as_tuples, method=method, top=top)
        fwh_ratio = FWHR_calc(corners)

        if show:
            print('The Facial-Width-Height ratio is: {}'.format(fwh_ratio))
            show_box(image, corners)
        else:
            return fwh_ratio
    else:
        if show:
            print("Picture is not suitable to calculate fwhr.")
            imshow(image)
        else:
            return None


def get_fwhr_bulk(folder_path, sort_by='filename'):
    """Calculates the facial-width-height ratio for all images in a folder and saves to a CSV file.

    Args:
        folder_path (str): The path to the folder containing the images.
        sort_by (str, optional): Specifies how the dataframe should be sorted. Accepted values are 'filename' and 'ratio'. Defaults to 'filename'.

    Returns:
        None. The function saves the dataframe as a CSV file in the same folder as the images.
    """
    # Create a list to store the image information
    image_data = []

    # Calculate the facial-width-height ratio for each image
    directory = os.fsencode(folder_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = folder_path + filename
            fwh_ratio = get_fwhr(image_path, url=False,
                                 show=False, method='average', top='eyelid')
            image_data.append([filename, fwh_ratio])

    # Convert the list to a dataframe
    df = pd.DataFrame(image_data, columns=['Filename', 'Ratio'])

    # Sort the dataframe
    if sort_by == 'filename':
        df.sort_values(by='Filename', inplace=True)
    elif sort_by == 'ratio':
        df.sort_values(by='Ratio', inplace=True)
    else:
        raise ValueError(
            f"Invalid value for 'sort_by': {sort_by}. Accepted values are 'filename' and 'ratio'.")

    # Save the dataframe as a CSV file
    csv_file = folder_path + 'fwhr_ratios.csv'
    df.to_csv(csv_file, index=False)


get_fwhr_bulk("./images/")
