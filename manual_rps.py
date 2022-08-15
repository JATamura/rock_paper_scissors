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
        if user_choice == "nothing":
            return 2
        elif user_choice == computer_choice:
            return 0
        elif (user_choice  == "rock" and computer_choice == "scissors") or (user_choice  == "paper" and computer_choice == "rock") or (user_choice  == "scissors" and computer_choice == "paper"):
            return 1
        else:
            return -1

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

if __name__ == "__main__":
    rps = manual_rps()
    rps.play()
