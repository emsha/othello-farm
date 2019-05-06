# trainer.py

from game import Game
from random_player import RandomPlayer
from net import Net
from net_player import NetPlayer
import random
from stateB import State
import torch
from conv_net import ConvNet
from trainer import Trainer
import random
from minimax_player import MinimaxPlayer
import sys

def play_n_games(n, player1, player2, show=False, showPercent=False, randomize=False):
	#we care about player 1's wins
	win_data = []
	win_count = 0
	for i in range(n):
		r = random.choice((0, 1))
		game = Game(player1, player2)
		if randomize and r:
			game = Game(player2, player1)
		score, win_pairs = game.play()
		if randomize and r:
			if score[0] == 2:
				win_count += 1
		else:
			if score[0]==1:
				win_count += 1
		if show:
			print(str(i) + ": " + str(score) + " -> {} moves".format(len(win_pairs)))
		for e in win_pairs:
			win_data.append(e)
	if showPercent:
		print("Win Rate: {}%".format(100*win_count/n))
	return win_data, win_count

def train_net_on_data(net, inputs, outputs, epochs):
	#net is a ConvNet, inputs and outputs are gonna be lists of mini batch stacked tensors
	netPlayer = NetPlayer(net)
	# print("initial test winrate:")
	for i in range(epochs):
		# print("--- Epoch {} ---".format(i))
		for j in range(len(inputs)):
			in_stack = inputs[j]
			out_stack = outputs[j]
			net.train_on_batch(in_stack, out_stack, 1, 0.005)
	
def test_net_random(net, n_games):
	# returns fraction of games won against random adversary
	return play_n_games(n_games, NetPlayer(net), RandomPlayer(), randomize=True)[1]/n_games
	
def test_net_minimax(net, n_games, depth):
	# returns fraction of games won against minimax adversary
	return play_n_games(n_games, NetPlayer(net), MinimaxPlayer(depth), randomize=True)[1]/n_games


def evaluate_net_fitness(net, n_random_games, n_minimax_games):
	# evaluates a fitness score, weighting more heavily winrates
	# against better adversaries. RETURNS score [0, 1]
	mini_w = .8
	rand_w = .2
	w_rand_score = test_net_random(net, n_random_games) * rand_w
	w_mini_score = test_net_minimax(net, n_minimax_games, 2) * mini_w
	return w_rand_score + w_mini_score

def gen_games_from_pop(nets, n_games):
	for net in nets:
		win_data = play_n_games(n_games, NetPlayer(net), RandomPlayer(), randomize=True, show=False)[0]
	return win_data

def train_pop_on_data(nets, win_data):
	# setup batches
	in_batch_list = []
	out_batch_list = []
	batch_size = 10
	epochs = 10
	inputs = [e[0] for e in win_data]
	outputs = [e[1] for e in win_data]
	c = 0
	for i in range(0, len(inputs), batch_size):
		try:
			in_batch_list.append(torch.stack(inputs[i:i+batch_size]))
			out_batch_list.append(torch.stack(outputs[i:i+batch_size]))
			c+=1
		except:
			pass
	#train nets
	for i, net in enumerate(nets):
		# print("made {} batches".format(c))
		# print("training...")
		# print("net {}: before: {}".format(i, evaluate_net_fitness(net, 10, 2)))
		print("training net {}".format(i))
		train_net_on_data(net, in_batch_list, out_batch_list, epochs)
		
		# print("net {} after: {}".format(i, evaluate_net_fitness(net, 10, 2)))
		

def main():
	# win_move_pairs 
	pop = []
	for _ in range(10):
		pop.append(ConvNet())
	print("populated with 10 net buddies")
	print("\n\ninitial evaluation--------:\n")
	for i, net in enumerate(pop): 
		print("Net {}: {}".format(i, test_net_random(net, 10)))

	print("done.")

	print("generating moves, 10 games each")
	win_data = gen_games_from_pop(pop, 20)
	print("{} moves generated".format(len(win_data)))
	print("training population...")
	# sys.stdout.write('\b \b')
	train_pop_on_data(pop, win_data)
	print("training complete")
	print("\n\nfinal evaluation--------:\n")

	for i, net in enumerate(pop): 
		print("Net {}: {}".format(i, test_net_random(net, 10)))

	print("done.")







	
	# play_n_games(10, MinimaxPlayer(2), RandomPlayer(), show=True, showPercent=True, randomize=False)








	

	# BATCH_SIZE = 10
	# NUM_GEN_GAMES = 1
	# NUM_POST_TRAIN_TEST_GAMES = 100
	# EPOCHS = 10

	# print("playing {} games to generate data".format(NUM_GEN_GAMES))
	# # win_data = play_n_games(NUM_GEN_GAMES, RandomPlayer(), RandomPlayer(), randomize=True, show=False)[0]
	# print("generated {} winning moves\n\n".format(len(win_data)))
	
	# # print(inputs)
	# # print(outputs)
	# net = ConvNet()
	# print(evaluate_net_fitness(net, 5, 5))

	# input("wait for itttt")

	# in_batch_list = []
	# out_batch_list = []
	# print("making batches")
	# c = 0
	# for i in range(0, len(inputs), BATCH_SIZE):
	# 	try:
	# 		in_batch_list.append(torch.stack(inputs[i:i+BATCH_SIZE]))
	# 		out_batch_list.append(torch.stack(outputs[i:i+BATCH_SIZE]))
	# 		c+=1
	# 	except:
	# 		pass
	# print("made {} batches boo".format(c))
	# print("training...")
	# train_net_on_data(net, in_batch_list, out_batch_list, EPOCHS)
	# print("done training :)")
	# play_n_games(NUM_POST_TRAIN_TEST_GAMES, NetPlayer(net), RandomPlayer(), randomize=True, showPercent=True)
	# test_net(net, 20)


	# in_stack = torch.stack(inputs[:10])
	# out_stack = torch.stack(outputs[:10])
	# net.train_on_batch(in_stack, out_stack, 1000)
	
	# first_moves = torch.max(out_stack, 1)
	# net_first_moves = net.forward_indices(in_stack)
	# print(first_moves[1])
	# print(net_first_moves[1])
	#test net


	# for i, e in enumerate(win_data):
	# 	s, m = e
	# 	print(i)
	# 	print(s)
	# 	print(State.output_tensor_to_coords(m))
	# 	print('\n\n\n')



if  __name__ =='__main__':main()