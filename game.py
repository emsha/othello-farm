import sys
from stateB import State
from player import Player
from random_player import RandomPlayer
import numpy as np
# from random_player import RandomPlayer

class Game():
	def __init__(self, player1, player2):
		self.state = State()
		self.player1 = player1
		self.player2 = player2
		self.winner = 0
		self.lastMove = None
		# self.moves = []
		self.game_over = False

	def play(self):
		ps = [self.player1, self.player2]
		i = 0
		states = []
		while not self.state.is_over():
			# self.state.pretty_print()
			# print(self.state.convert_to_padded_net_input_lists_dims())
			
			player = ps[i]
			self.state = player.play_move(self.state) 
			states.append(self.state)
			i = (i + 1) % 2
		# self.state.print_score()
		
		# get (saw_state, winning_move) pairs
		# loser_input_states = list(filter(lambda state: (state.to_move != self.state.score()[0]), states))
		# winner_input_states =list(filter(lambda state: (state.to_move == self.state.score()[0]), states)) 
		# win_moves_unfiltered = [l.last_move for l in loser_input_states]
		# winning_moves = list(filter(lambda move: (move is not None), win_moves_unfiltered))
		win_move_pairs = []
		for i, state in enumerate(states):
			#if state is a winner input state
			if state.to_move == self.state.score()[0]:
				try:
					move_from_state = states[i+1].last_move
					win_move_pairs.append((State.state_to_input_tensor(state, pad=False), State.coords_to_output_tensor(move_from_state)))
				except:
					pass
		# win move pairs is (state that the winner saw, coords of the move that they made)
		return self.state.score(), win_move_pairs


def main():
	# win_move_pairs 
	game0 = Game(RandomPlayer(), RandomPlayer())
	score, winmoves = game0.play()
	print(len(winmoves))
	for e in winmoves:
		print(e)

if  __name__ =='__main__':main()
