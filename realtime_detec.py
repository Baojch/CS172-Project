import cv2
import mediapipe as mp

# Initialize mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
mp_hands = mp.solutions.hands

# For static images
IMAGE_FILES = []
with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Shoe') as objectron_shoe, \
     mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Cup') as objectron_cup, \
     mp_hands.Hands(static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Objectron and Hands.
    results_shoe = objectron_shoe.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results_cup = objectron_cup.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    hand_results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    annotated_image = image.copy()

    # Draw shoe box landmarks.
    if results_shoe.detected_objects:
      print(f'Shoe box landmarks of {file}:')
      for detected_object in results_shoe.detected_objects:
        mp_drawing.draw_landmarks(
            annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
        mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                             detected_object.translation)

    # Draw cup box landmarks.
    if results_cup.detected_objects:
      print(f'Cup box landmarks of {file}:')
      for detected_object in results_cup.detected_objects:
        mp_drawing.draw_landmarks(
            annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
        mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                             detected_object.translation)

    # Draw hand landmarks.
    if hand_results.multi_hand_landmarks:
      for hand_landmarks in hand_results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input
cap = cv2.VideoCapture(0)

cv2.namedWindow('MediaPipe Objectron and Hands', cv2.WINDOW_NORMAL)  
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
fps = cap.get(cv2.CAP_PROP_FPS) 
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
writer = cv2.VideoWriter("1.mp4", fourcc, fps, (width, height))

print(f'width: {width}, height: {height}, fps: {fps}')
cv2.setWindowProperty('MediaPipe Objectron and Hands', width, height)

with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Shoe') as objectron_shoe, \
     mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Cup') as objectron_cup, \
     mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display.
    image = cv2.flip(image, 1)
    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results_shoe = objectron_shoe.process(image)
    results_cup = objectron_cup.process(image)
    hand_results = hands.process(image)

    # Draw the box landmarks on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw shoe box landmarks.
    if results_shoe.detected_objects:
        for detected_object in results_shoe.detected_objects:
            mp_drawing.draw_landmarks(
              image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)

    # Draw cup box landmarks.
    if results_cup.detected_objects:
        for detected_object in results_cup.detected_objects:
            mp_drawing.draw_landmarks(
              image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)

    # Draw hand landmarks on the image.
    if hand_results.multi_hand_landmarks:
      for hand_landmarks in hand_results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Objectron and Hands', image)
    # video save
    key = cv2.waitKey(24)
    writer.write(image) 

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

cv2.destroyAllWindows() 