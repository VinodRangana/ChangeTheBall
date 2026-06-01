import cv2
import os

from calculators.segmentationCalculator import SegmentationCalculator
from calculators.roundnessCalculator import RoundnessCalculator
from cricketBall import CricketBall  

class VisionInspector:
    def __init__(self):
        # Initialize the classes that hold the heavy math
        self.seg_calc = SegmentationCalculator()
        self.roundness_calc = RoundnessCalculator()
        # self.seam_calc = SeamCalculator()

    # --- THE WRAPPER FUNCTIONS ---

    def run_segmentation(self, ball: CricketBall):
        print(f"Starting Segmentation for {ball.ball_id}...")
        
        for view, raw_path in ball.raw_images.items():
            # 1. Load the data
            raw_img = cv2.imread(raw_path)
            
            # SAFETY CHECK: Did the image actually load?
            if raw_img is None:
                print(f"  -> ERROR: Could not read image at {raw_path}. Skipping.")
                continue  # Skips to the next image in the loop
            
            # 2. Hand it to the Calculator! 
            # FIX: Added 'self.' before seg_calc
            final_mask, isolated_ball = self.seg_calc.process_image(raw_img)
            
            # 3. Save the returned arrays to the hard drive
            folder_path = os.path.dirname(raw_path)
            mask_path = f"{folder_path}/mask_{view}.jpg"
            crop_path = f"{folder_path}/crop_{view}.jpg"
            
            cv2.imwrite(mask_path, final_mask)
            cv2.imwrite(crop_path, isolated_ball)
            
            # 4. Update the Data Object
            ball.masks[view] = mask_path
            ball.cropped_images[view] = crop_path


    def calculate_roundness(self, ball: CricketBall):
        print(f"Calculating Roundness for {ball.ball_id}...")
        
        scores = []
        
        # Loop through ALL masks (top, bottom, front, back, rough, smooth)
        for view, mask_path in ball.masks.items():
            
            # Load the mask in Grayscale mode
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask_img is not None:
                # Ask the calculator for the math
                score = self.roundness_calc.process_single_mask(mask_img)
                scores.append(score)
                print(f"    -> {view} view scored: {score:.3f}")
            else:
                print(f"    -> ERROR: Mask for {view} not found. Skipping.")
                
        # --- THE WORST-CASE RULE ---
        # If the ball passes on 5 sides but has a dent on the 6th, it must fail.
        # Therefore, the final score is the MINIMUM ratio found across all views.
        if scores:
            ball.roundness_score = min(scores)
        else:
            ball.roundness_score = 0.0
            
        print(f"-> Final Roundness Grade: {ball.roundness_score:.3f}")


    def calculate_final_grade(self, ball: CricketBall):
        # Your custom algorithm to weigh the final score
        weighted_score = (ball.seam_integrity_score * 0.4) + \
                         (ball.roughness_score * 0.4) + \
                         (ball.color_score * 0.2)
        
        ball.final_score = weighted_score
        
        # Set Status based on thresholds
        if ball.final_score > 75.0 and ball.roundness_score > 0.95:
            ball.status = "NO NEED TO CHANGE"
        else:
            ball.status = "CHANGE BALL"


    # --- THE MAIN EXECUTION PIPELINE ---
    
    def inspect(self, ball: CricketBall):
        # 1. Run Segmentation first
        self.run_segmentation(ball) 
        
        # 2. Run the Feature Wrappers (Assuming these are written elsewhere in your class)
        # self.calculate_roundness(ball)
        # self.calculate_seam(ball)
        # self.calculate_roughness(ball)
        # self.calculate_color(ball)
        
        # 3. Calculate Final Grade
        self.calculate_final_grade(ball)
        
        return ball