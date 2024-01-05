import cv2
import mediapipe as mp
import numpy as np

class Detector:
    
    def __init__(self, 
                mode: bool = False,
                number_hands: int = 2,
                model_complexity: int = 1,
                min_detec_confidence: float = 0.5,
                min_tracking_confidence: float = 0.5,
                ):
        
        #Required parameters to start the hands mediapipe   
        self.mode = mode
        self.max_number_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tracking_confidence

        #starting hands_model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                         self.max_number_hands,
                                         self.complexity,
                                         self.detection_con,
                                         self.tracking_con)
        
        #starting face_model
        self.face_mesh = mp.solutions.face_mesh
        self.model_face = self.face_mesh.FaceMesh()

        self.mp_draw = mp.solutions.drawing_utils

    #method that draws landmarks on the image 
    def draw_landmarks(self,
                   img: np.ndarray,
                   draw_hands: bool = True
                   ):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results_hands = self.hands.process(img_rgb)
        self.results_face = self.model_face.process(img_rgb)

        if self.results_face.multi_face_landmarks:
            for face in self.results_face.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    image = frame,
                    landmark_list = face,
                    connections = self.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )

        if self.results_hands.multi_hand_landmarks:
            for hand in self.results_hands.multi_hand_landmarks:
                if draw_hands:
                    self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
                    
        return img
        
if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    Detector = Detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = Detector.draw_landmarks(frame)

        frame = cv2.resize(frame, (1024,768))
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
