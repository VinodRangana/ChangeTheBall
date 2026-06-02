import cv2
import numpy as np
import math

class ColorCalculator:
    def __init__(self):
        # 1. Tolerances & Targets
        self.edge_erosion_ratio = 0.15
        
        # Visibility Targets for a perfect ball
        self.target_rel_brightness = 0.60  # Ball should be 60% as bright as the box
        self.target_saturation = 130.0     # Healthy red vibrancy (0-255 scale)
        
        # Critical Thresholds (If it drops below this, it fails)
        self.min_pass_shine = 0.50  # Shiny side must maintain 50% visibility
        self.min_pass_rough = 0.30  # Rough side can drop to 30% visibility
        
        # 2. Kernels
        self.text_connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))

    def create_safe_zone(self, img_array, circular_mask):
        """
        Reusing the exact same 'Blindfold' from Phase 4.
        Hides the perimeter, the white quarter-seam, and the logo so they 
        don't mathematically corrupt the red leather average.
        """
        height, width = img_array.shape[:2]
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # STEP 1: Erase Edges
        radius = width // 2
        erosion_pixels = int(radius * self.edge_erosion_ratio)
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_pixels, erosion_pixels))
        safe_mask = cv2.erode(circular_mask, erosion_kernel, iterations=1)

        # STEP 2: Erase Quarter-Seam
        edges = cv2.Canny(gray_img, 50, 150)
        edges_masked = cv2.bitwise_and(edges, edges, mask=safe_mask)
        min_line_length = int(width * 0.30)
        lines = cv2.HoughLinesP(edges_masked, 1, np.pi / 180, threshold=80, 
                                minLineLength=min_line_length, maxLineGap=20)
        
        if lines is not None:
            longest_line = max(lines, key=lambda l: math.hypot(l[0][2]-l[0][0], l[0][3]-l[0][1]))
            x1, y1, x2, y2 = longest_line[0]
            cv2.line(safe_mask, (x1, y1), (x2, y2), 0, thickness=20)

        # STEP 3: Erase Logo
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, morph_kernel)
        _, text_thresh = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)
        text_blobs = cv2.morphologyEx(text_thresh, cv2.MORPH_CLOSE, self.text_connect_kernel)
        contours, _ = cv2.findContours(text_blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if area > 500 and aspect_ratio > 1.5:
                cv2.rectangle(safe_mask, (x, y), (x+w, y+h), 0, -1)

        return safe_mask

    def calculate_visibility_metrics(self, img_array, safe_zone_mask, full_ball_mask):
        """
        Calculates the Relative Brightness (Ball vs Background) and Average Saturation.
        """
        # 1. Convert to Perceptual Color Space (HSV)
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        
        # 2. Extract the Background (The corners of the square crop)
        # Inverting the circular ball mask gives us the pure background box pixels!
        bg_mask = cv2.bitwise_not(full_ball_mask)
        
        # cv2.mean returns a tuple. We only want the first number [0].
        bg_brightness = cv2.mean(v, mask=bg_mask)[0]
        
        # 3. Extract the Ball's Valid Leather
        ball_brightness = cv2.mean(v, mask=safe_zone_mask)[0]
        ball_saturation = cv2.mean(s, mask=safe_zone_mask)[0]
        
        # Failsafe: Prevent division by zero if the background is completely pitch black
        bg_brightness = max(bg_brightness, 1.0)
        
        # 4. The Magic Ratio (Cancels out cloudy days vs sunny days)
        relative_brightness = ball_brightness / bg_brightness
        
        return relative_brightness, ball_saturation

    def grade_color(self, img_array, full_ball_mask):
        """
        The Master Orchestrator for Phase 5.
        Returns a Visibility Score from 0.0 to 1.0.
        """
        # 1. Build the blindfold
        safe_zone_mask = self.create_safe_zone(img_array, full_ball_mask)
        
        # 2. Get the metrics
        rel_bright, sat = self.calculate_visibility_metrics(img_array, safe_zone_mask, full_ball_mask)
        
        # 3. Convert raw metrics into a 0.0 - 1.0 Grade
        # Cap them at 1.0 so an ultra-bright ball doesn't break the math
        bright_score = min(1.0, rel_bright / self.target_rel_brightness)
        sat_score = min(1.0, sat / self.target_saturation)
        
        # Final Score is an equal balance of Brightness and Color Vibrancy
        visibility_score = (bright_score + sat_score) / 2.0
        
        print(f"    -> Metrics: Rel Brightness={rel_bright:.2f}, Saturation={sat:.1f} | Score={visibility_score:.2f}")
        
        return visibility_score