import csv
import os
import datetime

class DataLogger:
    def __init__(self, output_dir="data/output_logs"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create a unique filename based on the current time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.output_dir, f"session_{timestamp}.csv")
        self.initialized = False

    def log_frame(self, frame_count, thermal_landmarks, stimulus_data):
        """
        Saves frame data. 
        thermal_landmarks: np.array of 68 [x, y]
        stimulus_data: dict of {point_name: temperature}
        """
        # Prepare the row
        row = {
            "frame": frame_count,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Flatten the 68 landmarks into x0, y0, x1, y1... columns
        for i, (x, y) in enumerate(thermal_landmarks):
            row[f"lm_{i}_x"] = x
            row[f"lm_{i}_y"] = y
            
        # Add the temperature stimulus points
        row.update(stimulus_data)

        # Write to CSV
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self.initialized:
                writer.writeheader()
                self.initialized = True
            writer.writerow(row)
            