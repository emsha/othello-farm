import random
from stateB import State
from player import Player
# from monte_carlo_tree import MonteCarloTree, NetEvaluator

class NetPlayer(Player):

    def __init__(self, net):
        self.net = net

    def play_move(self, state, show_game=False):
        output = self.net(State.state_to_input_tensor(state).unsqueeze(0))
        # print("in play move:")
        # print(output)
        max_prob = -1000000
        best_state = None
        for next_state in state.legal_moves():
            if next_state.last_move == None:
                return next_state
            x,y = next_state.last_move
            prob = output[0][x+y*8]
            if prob > max_prob:
                max_prob = prob
                best_state = next_state
        return best_state

# def main():
#     m_player = MCTPlayer()
#     state = State()
#     state.pretty_print()
#     for _ in range(8):
#         state = m_player.play_move(state)
#         state.pretty_print()


# if  __name__ =='__main__':main()