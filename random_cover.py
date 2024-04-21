import cv2
import dlib
import os
import numpy as np
import random
from tqdm import tqdm
import logging

# Load the face detector
detector = dlib.get_frontal_face_detector()


def add_random_obstruction(img, face):
    # Extended shape choices: 0-rectangle, 1-circle, 2-ellipse, 3-triangle, 4-random polygon
    shape = random.choice([0, 1, 2, 3, 4])
    # Random color
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Random size as a proportion of the face size (30% to 40%)
    scale = random.uniform(0.3, 0.4)

    # Calculate the size and position of the obstruction
    width = int(face.width() * scale)
    height = int(face.height() * scale)
    x = random.randint(face.left(), face.right() - width)
    y = random.randint(face.top(), face.bottom() - height)

    if shape == 0:  # Rectangle
        cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
    elif shape == 1:  # Circle
        radius = min(width, height) // 2
        center = (x + width // 2, y + height // 2)
        cv2.circle(img, center, radius, color, -1)
    elif shape == 2:  # Ellipse
        center = (x + width // 2, y + height // 2)
        axes = (width // 2, height // 2)
        angle = 0
        startAngle = 0
        endAngle = 360
        cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, -1)
    elif shape == 3:  # Triangle
        points = np.array([
            [x + width // 2, y],
            [x, y + height],
            [x + width, y + height]], np.int32)
        cv2.fillPoly(img, [points], color)
    else:  # Random polygon
        num_vertices = random.randint(4, 6)  # Polygon with 4 to 6 vertices
        points = np.array([[
            random.randint(x, x + width), random.randint(y, y + height)
        ] for _ in range(num_vertices)], np.int32)
        cv2.fillPoly(img, [points], color)


def add_obstruction_to_faces(image_path, output_path):
    img = cv2.imread(image_path)

    if img is None:
        logging.error(f"Failed to load image: {image_path}. Skipping...")
        return

    faces = detector(img, 1)
    for face in faces:
        add_random_obstruction(img, face)

    cv2.imwrite(output_path, img)


def batch_process(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_names = os.listdir(input_dir)
    # Progress bar
    pbar = tqdm(total=len(img_names), desc='Processing images')

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        add_obstruction_to_faces(image_path, output_path)
        pbar.update(1)


# Set input and output directories
input_dir = './data/no_cover'
output_dir = './data/r_cover'

# Execute batch processing
batch_process(input_dir, output_dir)
