
import os
import json
import cv2
import traceback
import numpy as np

from camera_only import vision_only_distance_calculation

DATA_PATH = '../../data/'

def get_metadata() -> dict:
    with open(os.path.join(DATA_PATH, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    return metadata

def check_depth_accuracy(student_depth: float , sol_depth: float, pct_tol: float = .05) -> bool:
    err = pct_tol * sol_depth
    if abs(sol_depth - student_depth) <= err:
        return True
    return False
        

def check_angle_accuracy(student_angle: float, sol_angle: float, rad_tol: float = 0.08) -> bool:
    if abs(sol_angle - student_angle) <= rad_tol:
        return True
    return False


def test_vision_only() -> None:

    
    print("Running vision_only tests!")
    metadata = get_metadata()
    try:
        for image_name, values in metadata.items():
            image_data_path = os.path.join(DATA_PATH, image_name)
            fov = values['fov']
            student_depth, student_angle = vision_only_distance_calculation(cv2.imread(image_data_path), fov, 0.058*2)

            
            is_depth_correct = check_depth_accuracy(student_depth, values['vision_only_distance'])
            is_angle_correct = check_angle_accuracy(student_angle, values['vision_only_angle'])


            
            print(f"For {image_name}, you calculated the {'correct' if is_depth_correct else 'INCORRECT'} depth and the {'correct' if is_angle_correct else 'INCORRECT'} angle.")

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1] 
        function_name = tb.name
        line_number = tb.lineno
        raise Exception(f"{function_name}:{line_number} - {e}")


if __name__ == "__main__":
    test_vision_only()