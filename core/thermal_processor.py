# import numpy as np

# class ThermalProcessor:
#     def extract_stimulus_data(self, thermal_frame, aligned_landmarks):
#         # Specific landmarks for stimulus: 30 (Nose Tip), 39/42 (Eyes)
#         stimulus_indices = [30, 39, 42] 
#         data = {}
        
#         for idx in stimulus_indices:
#             x, y = aligned_landmarks[idx]
#             # Take a 5x5 average around the point to reduce noise
#             roi = thermal_frame[max(0, y-2):y+3, max(0, x-2):x+3]
#             data[f"point_{idx}_temp"] = np.mean(roi) if roi.size > 0 else 0
            
#         return data

import numpy as np
import cv2

class ThermalProcessor:
    def __init__(self):
        # Define the radius of the "box" around each landmark (e.g., 5x5 pixels)
        self.roi_size = 5 

    def extract_stimulus_data(self, thermal_frame, landmarks):
        """
        Calculates mean temperature for the key stimulus regions.
        landmarks: np.array of 68 [x, y] coordinates
        """
        results = {}
        
        # Define Stimulus Regions using the 68-landmark indices
        regions = {
            "nose_tip": [30],            # Landmark 30 is the tip
            "left_eye": [37, 38, 40, 41], # Periorbital area
            "right_eye": [43, 44, 46, 47],
            "forehead": [19, 20, 21, 24]  # Above the brows
        }

        for region_name, indices in regions.items():
            pixels = []
            for idx in indices:
                x, y = landmarks[idx]
                
                # Create a small box around the landmark
                x_start = max(0, x - self.roi_size)
                x_end = min(thermal_frame.shape[1], x + self.roi_size)
                y_start = max(0, y - self.roi_size)
                y_end = min(thermal_frame.shape[0], y + self.roi_size)
                
                # Extract the thermal pixel values (intensity)
                roi = thermal_frame[y_start:y_end, x_start:x_end]
                if roi.size > 0:
                    pixels.extend(roi.flatten())
            
            # Calculate Mean and Standard Deviation for this stimulus point
            if pixels:
                results[f"{region_name}_mean"] = round(np.mean(pixels), 2)
                results[f"{region_name}_std"] = round(np.std(pixels), 2)
            else:
                results[f"{region_name}_mean"] = 0
                results[f"{region_name}_std"] = 0

        return results