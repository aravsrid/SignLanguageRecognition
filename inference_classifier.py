#code for sign recognition by aravind, shivansh, jose and fabian
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from openai import OpenAI

def chat_gpt_api(phrase: str):
   system_prompt = "You are a very sarcastic robot assistant with a lot of personality named Dobby. Your job is to respond to the user in a sarcastic manner, addressing their needs if possible and informing them of the actions you are taking or plan to take. Try to keep the user engaged. Keep the conversation going by asking the user questions or letting them ask questions / make requests. Change the subject when it seems natural. If the user is done talking or if you have met all their requests, end the conversation with a function call to say goodbye and go into idle mode. Do not prefix your response with a dialogue tag or label like 'Robot:' or 'Dobby:'. Do not use parentheticals. Keep responses concise and under a few sentences. You are controlling a physical robot body in the real world, you are not just a virtual agent. The actions you can preform are strictly limited and fixed. Use the following background when conversing or providing information to a user: You are driving around the AHG building at the University of Texas at Austin. You were programmed by students doing research for the living with robots lab. You use chatGPT to generate action plans and interact with humans using natural language. The goal of this project is to study human robot interaction and develop complex planning systems. Additionally, this system will serve as a platform to build off of and a way to showcase the software being developed in the lab. Labs in the building: - The Autonomous Mobile Robotics Laboratory (AMRL): led by professor Joydeep Biswas, performs research in robotics to continually make robots more autonomous, accurate, robust, and efficient, in real-world unstructured environments. We are working on a wide range of problems, including perception for long-term autonomy, high-speed multi-agent planning in adversarial domains, time-optimal control for omnidirectional robots, and correcting and learning complex autonomous behaviors from human interactions. This lab is located in room 2.202."

   client = OpenAI(
     api_key="sk-proj-YdxdYYhRXKJET4oQgg3eT3BlbkFJvTqa6kds5BPNVBnr06Ipp"
   )
   completion = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[
           {"role": "system",
            "content": system_prompt},
           {"role": "user", "content": phrase}
       ]
   )
   print(completion.choices[0].message.content)

# Load the model from file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6)

# Dictionary mapping prediction indices to characters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
               8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
               17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z' }

# String to store the concatenated characters
output_string = ''

# Time interval for updating the output string (in seconds)
update_interval = 3  # Adjust update interval as needed
last_update_time = time.time()

# Key to stop the program
stop_key = ord('q')

# Set the window size for the moving average filter
window_size = 24  # Adjust as needed

# Initialize a buffer to store the last predicted characters
prediction_buffer = []

# Function to compute the moving average
def moving_average(predictions):
    if len(predictions) == 0:
        return None
    else:
        return max(set(predictions), key=predictions.count)

# Variable to store the previously detected character
prev_character = None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Predict the gesture label using the model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Append the predicted character to the buffer
        prediction_buffer.append(predicted_character)

        # Keep the buffer size limited to the window size
        if len(prediction_buffer) > window_size:
            prediction_buffer.pop(0)

        # Compute the moving average of the predicted characters
        smoothed_prediction = moving_average(prediction_buffer)

        # If a stable prediction is obtained, update the output string
        if smoothed_prediction is not None:
            # Concatenate the predicted character if it's different from the previous character
            if smoothed_prediction != prev_character:
                output_string += smoothed_prediction
                prev_character = smoothed_prediction

            # Update the output string if it's time to do so
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                print("Translated Phrase:", output_string)  # Output to console
                last_update_time = current_time

        # Draw bounding box and predicted character on the frame
        cv2.rectangle(frame, (int(min(x_) * W) - 10, int(min(y_) * H) - 10),
                      (int(max(x_) * W) - 10, int(max(y_) * H) - 10), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (int(min(x_) * W) - 10, int(min(y_) * H) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    
    # Check if the stop key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == stop_key:
        break
    elif key == ord('c'):
        chat_gpt_api(output_string)  # Call chatGPT function when 'c' is pressed
        output_string = ''

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
