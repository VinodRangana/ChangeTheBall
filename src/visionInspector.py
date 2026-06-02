import cv2
import os

from calculators.segmentationCalculator import SegmentationCalculator
from calculators.roundnessCalculator import RoundnessCalculator
from calculators.seamCalculator import SeamCalculator
from calculators.roughnessCalculator import RoughnessCalculator
from calculators.colorCalculator import ColorCalculator
from cricketBall import CricketBall  

class VisionInspector:
    def __init__(self):
        # Initialize the classes that hold the heavy math
        self.seg_calc = SegmentationCalculator()
        self.roundness_calc = RoundnessCalculator()
        self.seam_calc = SeamCalculator()
        self.roughness_calc = RoughnessCalculator()
        self.color_calc = ColorCalculator()

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


    def calculate_seam(self, ball: CricketBall):
        print(f"Calculating Seam Integrity for {ball.ball_id}...")
        
        seam_scores = {}
        
        # We only care about the 4 main views for the seam (ignore rough/smooth sides)
        for view in ['top', 'bottom', 'front', 'back']:
            # Grab the saved FILE PATHS from Phase 1
            crop_path = ball.cropped_images.get(view)
            mask_path = ball.masks.get(view)
            
            if not crop_path or not mask_path:
                print(f"    -> WARNING: Missing {view} view paths. Skipping.")
                continue
                
            # Load the images from the hard drive into NumPy arrays
            img_array = cv2.imread(crop_path)
            # Ensure the mask is loaded strictly as Grayscale (1 channel)
            mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
            
            if img_array is not None and mask_array is not None:
                # Run the Master Orchestrator from SeamCalculator
                score = self.seam_calc.grade_seam(img_array, mask_array)
                seam_scores[view] = score
                print(f"    -> {view} view scored: {score:.3f}")
            else:
                print(f"    -> ERROR: Could not load saved images for {view}.")

        # Veto Rule & Averages
        if not seam_scores:
            ball.seam_integrity_score = 0.0
            return

        lowest_side = min(seam_scores, key=seam_scores.get)
        lowest_score = seam_scores[lowest_side]
        
        if lowest_score < self.seam_calc.min_passing_score:
            print(f"CRITICAL REJECTION: The {lowest_side.upper()} seam failed.")
            ball.seam_integrity_score = 0.0
        else:
            total_score = sum(seam_scores.values())
            ball.seam_integrity_score = total_score / len(seam_scores)
            
        print(f"-> Final Seam Grade: {ball.seam_integrity_score:.3f}")


    def calculate_roughness(self, ball: CricketBall):
        """
        Wrapper function to evaluate the surface degradation on the rough/shine sides.
        """
        print(f"Calculating Surface Roughness for {ball.ball_id}...")
        
        roughness_scores = {}
        
        # We only care about the sides showing the "cheeks" of the ball for this phase
        for view in ['rough', 'shine']: 
            crop_path = ball.cropped_images.get(view)
            mask_path = ball.masks.get(view)
            
            if not crop_path or not mask_path:
                print(f"    -> WARNING: Missing {view} view paths. Skipping.")
                continue
                
            img_array = cv2.imread(crop_path)
            mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img_array is not None and mask_array is not None:
                # Call our Master Orchestrator!
                score = self.roughness_calc.grade_roughness(img_array, mask_array)
                roughness_scores[view] = score
                print(f"    -> {view} view scored: {score:.3f}")
            else:
                print(f"    -> ERROR: Could not load saved images for {view}.")

        # --- Aggregation Rule ---
        if not roughness_scores:
            ball.roughness_score = 0.0
            return

        # Check for Critical Failure (If one side is completely torn apart, the ball fails)
        lowest_side = min(roughness_scores, key=roughness_scores.get)
        lowest_score = roughness_scores[lowest_side]
        
        if lowest_score < self.roughness_calc.min_passing_score:
            print(f"CRITICAL REJECTION: The {lowest_side.upper()} surface is critically damaged.")
            ball.roughness_score = 0.0
        else:
            # Otherwise, average the health of both sides
            total_score = sum(roughness_scores.values())
            ball.roughness_score = total_score / len(roughness_scores)
            
        print(f"-> Final Surface Roughness Grade: {ball.roughness_score:.3f}")

    
    def calculate_color(self, ball: CricketBall):
        """
        Wrapper function to evaluate the overall Visibility (Color & Brightness).
        """
        print(f"Calculating Color Visibility for {ball.ball_id}...")
        
        color_scores = {}
        
        # Visibility heavily depends on the "cheeks" of the ball
        for view in ['rough', 'shine']: 
            crop_path = ball.cropped_images.get(view)
            mask_path = ball.masks.get(view)
            
            if not crop_path or not mask_path:
                print(f"    -> WARNING: Missing {view} view paths. Skipping.")
                continue
                
            img_array = cv2.imread(crop_path)
            # Ensure the full_ball_mask is strictly 1-channel Grayscale
            mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img_array is not None and mask_array is not None:
                score = self.color_calc.grade_color(img_array, mask_array)
                color_scores[view] = score
            else:
                print(f"    -> ERROR: Could not load saved images for {view}.")

        # --- Aggregation Rule ---
        if not color_scores:
            ball.color_score = 0.0
            return

        shine_score = color_scores.get('shine', 0.0)
        rough_score = color_scores.get('rough', 0.0)
        critical_failure = False

        # Apply the Umpire's Thresholds
        if shine_score < self.color_calc.min_pass_shine:
            print(f"CRITICAL REJECTION: The SHINY side is too dark/muddy ({shine_score:.2f}).")
            critical_failure = True
            
        if rough_score < self.color_calc.min_pass_rough:
            print(f"CRITICAL REJECTION: The ROUGH side is completely invisible ({rough_score:.2f}).")
            critical_failure = True

        if critical_failure:
            ball.color_score = 0.0
        else:
            # The ball's overall visibility is the average of both sides
            ball.color_score = (shine_score + rough_score) / 2.0
            
        print(f"-> Final Color Visibility Grade: {ball.color_score:.3f}")


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