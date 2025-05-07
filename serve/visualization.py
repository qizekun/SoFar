import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_vector_with_text(image, start_camera_point, direction, camera_matrix,
                          text="front", arrow_color=(0, 255, 0), arrow_size=15,
                          line_width=3, ring_radius=8, ring_width=2,
                          text_offset=(10, -40), font_size=20):
    """
    Draws a direction vector with a white circular ring at the start point and text next to the arrow on the image.

    :param image: Input image as a PIL Image object
    :param start_camera_point: The start point in camera coordinate system (x1, y1, z1)
    :param direction: Unit vector (x, y, z) representing the direction in the camera coordinate system
    :param camera_matrix: Camera intrinsic matrix, 3x3
    :param text: Text to display near the arrow
    :param arrow_color: Color of the arrow (RGB tuple)
    :param arrow_size: Size of the arrowhead
    :param line_width: Width of the arrow line
    :param ring_radius: Radius of the circular ring at the start point
    :param ring_width: Width of the ring's border
    :param text_offset: Offset for the text relative to the arrow endpoint (dx, dy)
    :param font_size: Font size for the text
    :return: The image with the vector, circular ring, and text drawn as a PIL Image object
    """
    # Ensure the direction vector is a unit vector
    direction = np.array(direction)
    direction /= np.linalg.norm(direction)

    # Project the start point to the image plane
    start_camera_point = np.array(start_camera_point)
    start_image_point = camera_matrix @ start_camera_point
    start_image_point /= start_image_point[2]  # Normalize to pixel coordinates
    start_pixel = (int(start_image_point[0]), int(start_image_point[1]))

    # Calculate the end point of the direction vector (in the camera coordinate system)
    scale = 0.2  # Set a scaling factor for visualizing the vector
    end_camera_point = start_camera_point + direction * scale
    end_image_point = camera_matrix @ end_camera_point
    end_image_point /= end_image_point[2]  # Normalize to pixel coordinates
    end_pixel = (int(end_image_point[0]), int(end_image_point[1]))

    # Draw the vector on the image
    image_with_vector = image.copy()
    draw = ImageDraw.Draw(image_with_vector, 'RGBA')
    draw.line([start_pixel, end_pixel], fill=arrow_color + (255,), width=line_width)

    # Calculate the arrowhead
    vector = np.array([end_pixel[0] - start_pixel[0], end_pixel[1] - start_pixel[1]])
    vector_length = np.linalg.norm(vector)
    if vector_length == 0:
        return image_with_vector  # Avoid division by zero
    unit_vector = vector / vector_length
    perpendicular = np.array([-unit_vector[1], unit_vector[0]])  # Perpendicular vector

    # Arrowhead points
    arrow_point1 = (
        end_pixel[0] - int(arrow_size * unit_vector[0] + arrow_size * perpendicular[0] / 2),
        end_pixel[1] - int(arrow_size * unit_vector[1] + arrow_size * perpendicular[1] / 2)
    )
    arrow_point2 = (
        end_pixel[0] - int(arrow_size * unit_vector[0] - arrow_size * perpendicular[0] / 2),
        end_pixel[1] - int(arrow_size * unit_vector[1] - arrow_size * perpendicular[1] / 2)
    )

    # Draw the arrowhead
    draw.polygon([end_pixel, arrow_point1, arrow_point2], fill=arrow_color + (255,))

    # Draw a white circular ring at the start point
    outer_circle = [
        start_pixel[0] - ring_radius, start_pixel[1] - ring_radius,
        start_pixel[0] + ring_radius, start_pixel[1] + ring_radius
    ]
    inner_circle = [
        start_pixel[0] - (ring_radius - ring_width), start_pixel[1] - (ring_radius - ring_width),
        start_pixel[0] + (ring_radius - ring_width), start_pixel[1] + (ring_radius - ring_width)
    ]
    draw.ellipse(outer_circle, outline=(255, 255, 255, 255), width=ring_width)
    draw.ellipse(inner_circle, fill=(0, 0, 0, 0))  # Transparent inner area

    # Add text near the arrow endpoint using the built-in font
    text_position = (end_pixel[0] + text_offset[0], end_pixel[1] + text_offset[1])
    font = ImageFont.load_default(size=font_size)  # Use the default font
    draw.text(text_position, text, fill=(255, 255, 255, 255), font=font)

    return image_with_vector


def interpolate_frames(frame1, frame2, num_interpolated_frames=1):
    # Generate interpolated frames between frame1 and frame2
    interpolated_frames = []
    for i in range(1, num_interpolated_frames + 1):
        alpha = i / (num_interpolated_frames + 1)
        interpolated_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        interpolated_frames.append(interpolated_frame)
    return interpolated_frames


def save_images_as_video(image_list, output_video_path, frame_rate=30, interpolated_frame_rate=2):
    # Get the dimensions of the first image
    frame_height, frame_width = image_list[0].size[1], image_list[0].size[0]

    # Create a VideoWriter object to save the video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    # Convert images from PIL to NumPy arrays (OpenCV format)
    frames = [np.array(img) for img in image_list]

    # If the image is RGB, convert it to BGR for OpenCV compatibility
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame for frame in frames]

    # Write the first frame
    video_writer.write(frames[0])

    # Interpolate frames between each pair of images
    for i in range(len(frames) - 1):
        video_writer.write(frames[i])  # Write the current frame
        interpolated_frames = interpolate_frames(frames[i], frames[i + 1], interpolated_frame_rate)
        for interpolated_frame in interpolated_frames:
            video_writer.write(interpolated_frame)  # Write interpolated frames
        video_writer.write(frames[i + 1])  # Write the next frame

    # Release the VideoWriter object
    video_writer.release()


if __name__ == "__main__":
    image = Image.open('assets/navigation.png').convert("RGB")
    start_camera_point = (0.922, 0.030, 2.378)  # Starting point in camera coordinate system
    direction = (-0.882, 0.071, -0.466)  # Direction vector in camera coordinate system
    camera_matrix = np.array([
        [385.327, 0, 322.151],
        [0, 384.862, 246.106],
        [0, 0, 1]
    ])

    result_image = draw_vector_with_text(image, start_camera_point, direction, camera_matrix, text="front")
    result_image.show()
