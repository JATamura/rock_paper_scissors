# Computer Vision Rock Paper Scissors Project Documentation

> AiCore project documentation for replicating the game Rock-Paper-Scissors using computer vision. The Teachable-Machine website was used to train the model while Python was used to use the model to create the game.

## Milestone 1: Set up the environment 

## Milestone 2: Create the model

- Using Teachable-Machine, a keras model was created. It used images taken with myself showing the following options to the camera: rock, paper, scissors, nothing. The 'nothing' class was when no movement was made that was significant to the game itself. This model would be loaded into an Anaconda virtual environment and used to determine what action was chosen from the image taken by the computer's webcam.

## Milestone 3: Install the dependencies

- By using Anaconda prompt, a virtual conda environment was created. From there, the libraries needed to implement use the keras model downloaded from Teachable-Machine. These libraries are opencv-python, tensorflow, and ipykernel. These enable image/array manipulation and allows the keras model to be loaded up and used to predict the user's action using their camera. By running the given code, the conda environment uses the webcam to capture the user's motions and predict what choice they have made.

```python
import cv2
from keras.models import load_model
import numpy as np
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True: 
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    cv2.imshow('frame', frame)
    # Press q to close the window
    print(prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
```

### Figure 1: Using the webcam to detect motion

```
1/1 [==============================] - 0s 21ms/step
[[2.9242108e-05 7.2876744e-02 9.3563489e-07 9.2709309e-01]]
1/1 [==============================] - 0s 19ms/step
[[2.1573952e-04 2.3099964e-02 1.9928730e-05 9.7666442e-01]]
1/1 [==============================] - 0s 20ms/step
[[0.03049408 0.65205944 0.15938608 0.15806037]]
```

- As can be seen in Figure 1, the last action which was an open-hand motion is heavily weighted to be perdicted as paper as it has the heighest value (0.652..). This model will be used later when implementing a fully computer vision based rock paper scissors game.

## Milestons 4: Create a Rock-Paper-Scissors game

- First, a manual rock paper scissors game must be created in which the user inputs are text based in order to smoothly implement the model into the logic. This was fairly simple as the class manual_rps was innitialised to have a list with 3 option: rock, paper, and scissors.

```python
class manual_rps:

    def __init__(self):
        self.options = ["rock", "paper", "scissors"]
```

- First, the computer would make a random choice (the random extension was imported and used) from the options list above. Then the user would be given an input option to choose their hand. This option would need to be a vali choice (alphabetical and one of the 3 options in the list above) or else it would print *"Please choose rock, paper, or scissors"*.

```python
def get_computer_choice(self):
    return random.choice(self.options)

def get_user_choice(self):
    choice = ""
    while choice == "":
        temp = input("Rock, paper, scissors, shoot! ")
        if temp.isalpha() and temp.lower() in self.options:
            choice = temp
        else:
            print("Please choose rock, paper, or scissors")
    return choice
```

- Once both choices were selected, a get__winner method would use a set of if-elif-else statements to find the winner.

```python
def get_winner(self, computer_choice , user_choice):
    if computer_choice  == "rock":
        if user_choice == "rock":
            return 0
        elif user_choice == "paper":
            return 1
        else:
            return -1
    elif computer_choice  == "paper":
        if user_choice == "rock":
            return -1
        elif user_choice == "paper":
            return 0
        else:
            return 1
    else:
        if user_choice == "rock":
            return 1
        elif user_choice == "paper":
            return -1
        else:
            return 0
```
- Once the winner is selected, the program would print *"You won!"*, *"You lost!"*, or *"It was a tie!"* dependning on the outcome.

## Milestone 5: Use the camera to play Rock-Paper-Scissors

- By adjusting the code given in milestone 3, it could be inserted into the rock paper scissors game to use the webcam instead of a text based input. The model would be loaded outside of this get_prediction method and instead would be loaded as a class variabel when initializing the game.

```python
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
```

- The first while loop in the turns the camera on and keeps it on during down time between rounds of the game. When the 'r' key is pressed, a 4 seconds countdown (3, 2, 1, 0) will be started and displayed on the webcam display, upon which the image captured by the webcam at the end of the countdown would be used as the image to be parsed on into the model. The time.time() method was used to store the time that the countdown was started and in the seconds while loop, an if statement was used to print said countdown. After predicting the action presented in the image, *"press c to continue"* was printed. When the 'c' key is pressed, the prediction is used to get the user choice and find the winner.
