import os
from dataclasses import dataclass, field
from typing import Dict
import cv2

@dataclass
class CricketBall:
    ball_id: str
    overs_bowled: int
    
    # Dictionaries to store the String paths
    raw_images: Dict[str, str] = field(default_factory=dict)
    cropped_images: Dict[str, str] = field(default_factory=dict)
    masks: Dict[str, str] = field(default_factory=dict)
    
    # Feature Scores
    roundness_score: float = 0.0
    seam_integrity_score: float = 0.0
    roughness_score: float = 0.0
    color_score: float = 0.0
    
    # Final Output
    final_score: float = 0.0
    status: str = "UNGRADED"

    #to create ball object
    @classmethod
    def from_folder(cls, ball_id: str, folder_path: str, overs: int):
        print(f"Loading data for {ball_id}...")
        
        # 'cls' is a Python keyword that refers to the class itself (CricketBall).
        # We create the empty object first.
        new_ball = cls(ball_id=ball_id, overs_bowled=overs)
        
        expected_views = ['top', 'bottom', 'front', 'back', 'rough', 'smooth']
        
        for view in expected_views:
            file_path = f"{folder_path}/raw_{view}.jpg"
            if os.path.exists(file_path):
                new_ball.raw_images[view] = file_path
            else:
                print(f"  -> WARNING: Missing {view} image.")
                
        return new_ball

    # Lazy-Loading Display Function
    def display_result(self, view_name: str):
        print(f"[{self.ball_id}] Status: {self.status} | Final Score: {self.final_score}")
        
        # Example: Safely check if the image path exists before trying to open it
        if view_name in self.raw_images:
            img = cv2.imread(self.raw_images[view_name])
            cv2.imshow(f"{self.ball_id} - {view_name}", img)
            cv2.waitKey(0)
        else:
            print(f"Error: No image found for view '{view_name}'")