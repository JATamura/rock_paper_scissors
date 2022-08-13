import cv2
from keras.models import load_model
import numpy as np
import random
import time

class camera_rps:

    def __init__(self):
        self.options = ["rock", "paper", "scissors", "nothing"]
        self.model = load_model('keras_model.h5')
        self.user_wins = 0
        self.computer_wins = 0

    def get_prediction(self):
        cap = cv2.VideoCapture(0)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        print("press r when ready")
        while True: 
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                break
        countdown = 4
        predict_once = 0
        start = time.time()
        while True: 
            ret, frame = cap.read()
            if countdown > 1:
                cv2.putText(frame, str(countdown - 1), (236, 316), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 255, 255), 20)
            cv2.imshow('frame', frame)  
            if time.time() > start - countdown + 5 and countdown > 0:
                if countdown > 1:
                    print(countdown - 1)
                countdown -= 1
            elif countdown == 0 and predict_once == 0:
                resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
                image_np = np.array(resized_frame)
                normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
                data[0] = normalized_image
                prediction = self.model.predict(data)
                predict_once = 1
                print("press c to continue")
            if cv2.waitKey(1) & 0xFF == ord('c') and countdown == 0:
                break
        return prediction

    def get_computer_choice(self):
        return random.choice(self.options[:-1])

    def get_user_choice(self, prediction):
        choice = np.argmax(prediction)
        print("you chose " + self.options[choice])
        return self.options[choice]

    def get_winner(self, computer_choice , user_choice):
        if user_choice  == "rock":
            if computer_choice == "rock":
                return 0
            elif computer_choice == "paper":
                return -1
            else:
                return 1
        elif user_choice  == "paper":
            if computer_choice == "rock":
                return 1
            elif computer_choice == "paper":
                return 0
            else:
                return -1
        elif user_choice  == "scissors":
            if computer_choice == "rock":
                return -1
            elif computer_choice == "paper":
                return 1
            else:
                return 0
        else:
            return 2

    def play(self):
        while self.user_wins < 3 and self.computer_wins < 3:
            computer_choice = self.get_computer_choice()
            user_choice = self.get_user_choice(self.get_prediction())
            winner = self.get_winner(computer_choice, user_choice)
            print(user_choice.lower() + " : " + computer_choice.lower())
            if winner == 1:
                self.user_wins += 1
                print("You won!")
            elif winner == -1:
                print("You lost!")
                self.computer_wins += 1
            elif winner == 0: 
                print("It was a tie!")
            else:
                print("No choice was detected.")
            print("The score is " + str(self.user_wins) + " : " + str(self.computer_wins))
        if self.user_wins == 3:
            print("You won the match!")
        else:
            print("You lost the match!")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rps = camera_rps()
    rps.play()
