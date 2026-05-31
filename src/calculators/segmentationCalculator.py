import cv2
import numpy as np

class SegmentationCalculator:
    def __init__(self):
        # The Morphological kernel remains static, as it is based on physical pixel size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


    def _get_dynamic_cyan_bounds(self, hsv_img):
        """
        Dynamically calculates the exact hue of the background for THIS specific photo.
        """
        # 1. Extract the Hue channel
        hue_channel = hsv_img[:, :, 0]
        
        # 2. Calculate the Histogram for the Hue channel (0-179)
        hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
        
        # 3. SAFETY NET: Only look for the peak inside the "Cyan/Blue" spectrum (Hue 60 to 120)
        # This prevents the algorithm from accidentally locking onto the red ball.
        cyan_region = hist[60:120]
        
        # 4. Find the exact peak in that region and calculate the true Hue number
        peak_offset = np.argmax(cyan_region)
        peak_hue = 60 + peak_offset 
        
        # 5. Create dynamic bounds (+/- 15) around the exact lighting of this photo
        # We keep Sat/Val wide open to handle natural shadows and bright spots
        lower_cyan = np.array([max(0, peak_hue - 15), 30, 30])
        upper_cyan = np.array([min(179, peak_hue + 15), 255, 255])
        
        print(f"    -> Dynamic Background Detected at Hue: {peak_hue}")
        
        return lower_cyan, upper_cyan


    def process_image(self, raw_img_array):
        # 1. Convert to HSV
        hsv_img = cv2.cvtColor(raw_img_array, cv2.COLOR_BGR2HSV)
        
        # 2. AUTOMATION: Get the perfect thresholds for this exact image
        lower_cyan, upper_cyan = self._get_dynamic_cyan_bounds(hsv_img)
        
        # 3. Threshold the Background
        cyan_bg_mask = cv2.inRange(hsv_img, lower_cyan, upper_cyan)
        
        # 4. Invert to get the Raw Ball Mask
        raw_ball_mask = cv2.bitwise_not(cyan_bg_mask)
        
        # 5. Morphological Cleaning (Close then Open)
        cleaned_mask = cv2.morphologyEx(raw_ball_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        
        # 6. Find the Largest Contour (Protecting the physical edge)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_mask = np.zeros_like(cleaned_mask)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=-1)
            
        # 7. Create the Isolated Ball
        isolated_ball = cv2.bitwise_and(raw_img_array, raw_img_array, mask=final_mask)
        
        return final_mask, isolated_ball