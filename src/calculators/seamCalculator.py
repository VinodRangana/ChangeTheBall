import cv2
import numpy as np
from skimage.filters import frangi

class SeamCalculator:
    def __init__(self):
        # 20-pixel kernel for the Safety Boundary erosion
        self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        
        # Crop percentages
        self.top_bottom_crop_ratio = 0.15  # 15% deep crop
        self.roi_width_ratio = 0.30        # middle 30% for the safe box

    def segment_seam(self, img_array, mask_array):
        """
        Executes the 8-Step Pre-Processing Pipeline to isolate, straighten, 
        and crop the cricket ball seam.
        """
        height, width = img_array.shape[:2]

        # ==========================================
        # STEP 1: The Safety Boundary (Mask Erosion)
        # ==========================================
        # Shrink the mask to hide the sharp outer edge of the ball
        eroded_mask = cv2.erode(mask_array, self.erosion_kernel, iterations=1)

        # ==========================================
        # STEP 2: The Frangi Ridge Scan
        # ==========================================
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Apply the eroded mask to the grayscale image so Frangi ignores the background
        masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=eroded_mask)
        
        # Run Frangi Filter (black_ridges=False because our threads are bright white)
        # This returns a float matrix of the glowing heatmap
        frangi_heatmap = frangi(masked_gray, black_ridges=False)

        # ==========================================
        # STEP 3: Binarize the Heatmap (Otsu)
        # ==========================================
        # Normalize the float heatmap to standard 0-255 pixel format
        if np.max(frangi_heatmap) > 0:
            frangi_norm = (frangi_heatmap / np.max(frangi_heatmap) * 255).astype(np.uint8)
        else:
            frangi_norm = np.zeros_like(gray_img)

        # Apply Otsu's Thresholding to get pure 1s and 0s
        _, binary_seam = cv2.threshold(frangi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ==========================================
        # STEP 4: Measure the True Tilt (Hough Voting)
        # ==========================================
        # Break the white pixels into tiny line segments
        lines = cv2.HoughLinesP(binary_seam, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
        
        median_angle_degrees = 0.0
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle in radians, then convert to degrees
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # We want the angle relative to the Y-axis (Vertical)
                # A perfectly vertical line is 90 degrees or -90 degrees in OpenCV math
                if angle < 0:
                    angle += 180
                angles.append(angle - 90) # Shift so 0 degrees means perfectly vertical
                
            # The Voting System: Take the median to ignore bent/smashed threads
            median_angle_degrees = np.median(angles)

        # ==========================================
        # STEP 5: Auto-Rotate (The Straightening)
        # ==========================================
        # Get the center of the image to spin around
        center = (width // 2, height // 2)
        
        # Create the rotation math matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle_degrees, 1.0)
        
        # Spin the binary seam image (distorted tips move to the absolute top/bottom)
        rotated_seam = cv2.warpAffine(binary_seam, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)

        # ==========================================
        # STEP 6: The Deep Crop (Fixing Distortion)
        # ==========================================
        # Calculate exactly how many pixels 15% is
        crop_y = int(height * self.top_bottom_crop_ratio)
        
        # Slice off the top and bottom poles!
        # (We keep the X-axis full width for now)
        flat_seam = rotated_seam[crop_y : height - crop_y, :]

        # ==========================================
        # STEP 7: Lock the Dynamic ROI (Remove Scratches)
        # ==========================================
        new_height, new_width = flat_seam.shape[:2]
        
        # Find the coordinates of all remaining white pixels
        white_pixels = np.column_stack(np.where(flat_seam > 0))
        
        if len(white_pixels) > 0:
            # Calculate the Center of Mass on the X-axis (Width)
            center_x = int(np.mean(white_pixels[:, 1]))
        else:
            # Failsafe if the image is blank
            center_x = new_width // 2 
            
        # Define the Safe Box (e.g., 15% left of center, 15% right of center)
        box_half_width = int(new_width * (self.roi_width_ratio / 2))
        
        safe_left = max(0, center_x - box_half_width)
        safe_right = min(new_width, center_x + box_half_width)
        
        # Create a final black mask, and copy ONLY the pixels inside the safe box
        final_clean_seam = np.zeros_like(flat_seam)
        final_clean_seam[:, safe_left:safe_right] = flat_seam[:, safe_left:safe_right]

        # ==========================================
        # STEP 8: Delivery
        # ==========================================
        # We return a perfectly straight, cropped, binarized image containing ONLY the seam
        return final_clean_seam