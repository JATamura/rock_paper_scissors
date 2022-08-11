import random

class manual_rps:

    def __init__(self):
        self.options = ["rock", "paper", "scissors"]

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

    def play(self):
        computer_choice = self.get_computer_choice()
        user_choice = self.get_user_choice()
        winner = self.get_winner(computer_choice, user_choice)
        print(user_choice.lower() + " : " + computer_choice.lower())
        if winner == 1:
            print("You won!")
        elif winner == -1:
            print("You lost!")
        else: 
            print("It was a tie!")

rps = manual_rps()
rps.play()