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
from os import listdir
from os.path import isfile, join
import progressbar
from time import sleep

def play_n_games(n, player1, player2, show=False, showPercent=False, randomize=False):
    #we care about player 1's wins
    win_data = []
    win_count = 0
    for i in range(n):
        # print(".")
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

def gen_games_from_pop(nets, n_games, showbar=False):
    bar = None
    if showbar:
        bar = progressbar.ProgressBar(maxval=len(nets), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.SimpleProgress()])
        bar.start()
    for i, net in enumerate(nets):
        # print("net {} playing {} games".format(i, n_games))
        if showbar: bar.update(i+1)
        win_data = play_n_games(n_games, NetPlayer(net), RandomPlayer(), randomize=True, show=False)[0]
    if showbar: bar.finish()
    return win_data

def train_pop_on_data(nets, win_data, showbar=False):
    # setup batches
    in_batch_list = []
    out_batch_list = []
    batch_size = 10
    epochs = 10
    inputs = [e[0] for e in win_data]
    outputs = [e[1] for e in win_data]
    c = 0
    bar = None
    if showbar: 
        bar = progressbar.ProgressBar(maxval=len(nets), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.SimpleProgress()])
        bar.start()
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
        # print("training net {}".format(i))
        
        train_net_on_data(net, in_batch_list, out_batch_list, epochs)
        if showbar: bar.update(i+1)
        # print("net {} after: {}".format(i, evaluate_net_fitness(net, 10, 2)))
    if showbar: bar.finish()
def save_pop(pop, folder_path):
    for i, net in enumerate(pop):
        p = folder_path+"net"+str(i)+".pt"
        print(p)
        torch.save(net, p)

def load_pop(dirpath):
    p = []
    fnames = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    print(fnames)
    for f in fnames:
        model = torch.load(dirpath+f)
        model.eval()
        p.append(model)
    return p

def gen_pop(n):
    return [ConvNet() for _ in range(n)]

def eval_pop_random(pop, n, showbar=False):
    '''
    pop: population (list of nets)
    n: number of test games to be played
    returns -> (population avg winrate, [list of net's individal winrates in test])
    '''
    # print("hey")
    s = 0
    res = [0]*len(pop)
    # if show:
    #     print("***Evaluating population of {} nets on {} games each against random player".format(len(pop), n))
    bar=None
    if showbar:
        bar = progressbar.ProgressBar(maxval=len(pop), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.SimpleProgress()])
        bar.start()

    for i, net in enumerate(pop):
        winrate = test_net_random(net, n)
        s += winrate 
        res[i] = winrate
        # print("Net {}: {}".format(i, winrate))
        if bar:
            bar.update(i+1)
    avg_s = s/len(pop)
    # if show:
        # print("population winrate = {}".format(avg_s))
    if bar:
        bar.finish()
    return avg_s, res

def repopulate(pop, results, top_n):
    '''
    edits pop in place
    preserves the top top_n nets after n_games games against randomplayer
    and repopulates rest of spots with mutations of the top n nets
    '''
    
    top_i_list = sorted(range(len(results)), key=lambda i: results[i])[-(top_n):]
    # print(top_i_list)
    new_pop = []
    # add top n to new population
    for top_index in top_i_list:
        new_pop.append(pop[top_index])
    # fill in rest with mutations of top n
    for i in range(top_n, len(pop)):
        new_buddy = new_pop[i%top_n].clone(mutations=True)
        new_pop.append(new_buddy)
    # change pop in place
    pop = new_pop




def main():
    pop = gen_pop(7)
    GENERATIONS = 10
    N_TEST_GAMES = 50
    N_TRAIN_GAMES = 75
    EPOCHS = 18
    LEARNING_RATE = .001
    TOP_N = 3

    for g in range(GENERATIONS):
        print("\n\nGEN {}".format(g))
        
        # train:
        print("Generating {} games for training...".format(N_TRAIN_GAMES* len(pop)))
        windata = gen_games_from_pop(pop, N_TRAIN_GAMES, showbar=True)
        print("    Training pop on data ({} moves)...".format(len(windata)))
        train_pop_on_data(pop, windata, showbar=True)
        print("    Evaluating population against random mover...")
        
        # evaluate:
        score, results = eval_pop_random(pop, N_TEST_GAMES, showbar=True)
        print("\n        Score: {}\n    Results:{}\n".format(score, results))
        
        # repopulate
        print("    Repopulating with top {} nets".format(TOP_N))
        repopulate(pop, results, TOP_N)
    print(":)")
    folder_path = "/Users/maxshashoua/Documents/Developer/othellofarm/population0/"
    save_pop(pop, folder_path)
    


if  __name__ =='__main__':main()