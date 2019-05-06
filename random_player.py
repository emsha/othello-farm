import random
from stateB import State
from player import Player

class RandomPlayer(Player):

    def play_move(self, state, show_game=False):
    	# returns None if no legal moves (pass)
        newstate = random.choice(state.legal_moves())
        if show_game:
            print("\n\n\n")
            newstate.pretty_print()
        return newstate

def main():
    r_player = RandomPlayer()
    state = State()
    state.pretty_print()
    for _ in range(100):
        state = r_player.play_move(state, True) 



if  __name__ =='__main__':main()
