class CricketBall:
    def __init__(self, id, overs_bowled,top, front, s_side ,r_side , back, bottom ):
        self.ball_id = id
        self.overs_bowled = overs_bowled

        self.org_img_path = {
            "top" : top,
            "front" : front,
            "shine-side" : s_side,
            "rough-side" : r_side,
            "back" : back,
            "bottom" :bottom
        }

        self.enhanced_img = {
            "top" : None,
            "front" : None,
            "side" : None
        }
        
        self.sphericity_error = 0.0
        self.roughness_mean = 0.0
        self.seam_gap_pixels = 0.0
        self.is_fit_for_play = True