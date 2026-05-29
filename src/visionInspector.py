from calculators.roundness_calculator import RoundnessCalculator
from calculators.seam_calculator import SeamCalculator
# import other calculators...

class VisionInspector:
    def __init__(self):
        # Initialize the classes that hold the heavy math
        self.roundness_calc = RoundnessCalculator()
        self.seam_calc = SeamCalculator()

    # --- THE WRAPPER FUNCTIONS ---

    def calculate_roundness(self, ball):
        # Extract only the 3 orthogonal masks needed
        keys_to_check = ['top', 'front', 'rough']
        scores = []
        
        for key in keys_to_check:
            mask_path = ball.masks.get(key)
            if mask_path:
                # Call the heavy algorithm from the calculator class
                score = self.roundness_calc.process_single_mask(mask_path)
                scores.append(score)
                
        # Aggregate the scores (e.g., take the average or the worst case)
        ball.roundness_score = sum(scores) / len(scores) if scores else 0.0

    def calculate_seam(self, ball):
        # Extract the 4 views that contain the seam
        keys_to_check = ['top', 'front', 'bottom', 'back']
        scores = []
        
        for key in keys_to_check:
            crop_path = ball.cropped_images.get(key)
            if crop_path:
                score = self.seam_calc.process_single_seam(crop_path)
                scores.append(score)
                
        ball.seam_integrity_score = sum(scores) / len(scores) if scores else 0.0

    def calculate_final_grade(self, ball):
        # Your custom algorithm to weigh the final score
        # Example: Seam is 40% of the grade, Roughness is 40%, Color is 20%
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
    
    def inspect(self, ball):
        # 1. Run Segmentation first (Fills up the mask and cropped dictionaries)
        # self.run_segmentation(ball) 
        
        # 2. Run the Feature Wrappers
        self.calculate_roundness(ball)
        self.calculate_seam(ball)
        # self.calculate_roughness(ball)
        # self.calculate_color(ball)
        
        # 3. Calculate Final Grade
        self.calculate_final_grade(ball)
        
        return ball