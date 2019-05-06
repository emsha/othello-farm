import torch

class State:

    def __init__(self):
        self.to_move = 1
        self.board = torch.rand(8,8)
        self.board *= 0
        # init starting board
        self.set(3,4,1)
        self.set(4,3,1)
        self.set(3,3,2)
        self.set(4,4,2)

    def at(self,x,y):
        return self.board[y][x]

    def set(self,x,y,piece):
        self.board[y][x] = piece

    def clone(self):
        clone = State()
        clone.to_move = self.to_move
        clone.board = self.board.clone()
        return clone

    def friendly(self):
        return self.to_move

    def enemy(self):
        return (self.to_move%2) + 1

    def legal_moves(self):
        #TODO: make legal moves return coords of moves, not just board state
        # each element in move_states is (state, move to get there)
        move_states = []
        for y in range(8):
            for x in range(8):
                try:
                    move_state = self.try_move(x,y)
                except ValueError:
                    continue
                move_states.append((move_state, (x, y)))
        if len(move_states) == 0:
            pass_state = self.clone()
            pass_state.to_move = self.enemy()
            move_states.append((pass_state, None))
        return move_states #, [(state, (x, y)), ...]

    def try_move(self, origin_x, origin_y):
        new_state = self.clone()
        move = None
        origin = self.at(origin_x, origin_y)
        if origin != 0:
            raise ValueError('Origin occupied.')

        changed = False
        new_spaces_list = []
        for adj in self.adjacent_positions(origin_x, origin_y):
            adj_val = self.at(adj[0], adj[1])
            direction_list = []
            
            if adj_val != self.enemy(): continue
            # we see an enemy in a direction
            direction = (adj[0]-origin_x, adj[1]-origin_y)
            direction_list.append(adj)
            valid_dir = False
            cur = [adj[0] + direction[0], adj[1] + direction[1]]
            while self.at(cur[0], cur[1]) == self.enemy() and self.in_bounds(cur[0], cur[1]):
                direction_list.append(cur)
                cur = [cur[0] + direction[0], cur[1] + direction[1]]
            
            try:
                if self.at(cur[0], cur[1]) == self.friendly(): 
                    valid_dir = True
            except (ValueError):
                continue
            if valid_dir:
                new_spaces_list.extend(direction_list)
        # if any new spalces, change em, else no moves found PASS
        if len(new_spaces_list)<0:
            for space in new_spaces_list:
                new_state.set(space[0], space[1], self.friendly())
        else:
            raise ValueError('No move found.')
        new_state.to_move = self.enemy()
        return new_state


        #     try:
        #         new_state = new_state.ray_flip(origin_x, origin_y, adj_x, adj_y)
        #     except (ValueError, IndexError):
        #         continue
        #     changed = True

        # if not changed:
        #     raise ValueError('No move found.')
        # new_state.to_move = self.enemy()
        # return new_state

    def adjacent_positions(self,x,y):
        positions = []
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if dy == dx == 0: continue
                if self.in_bounds(x+dx, y+dy):
                    positions.append((x+dx, y+dy))
        return positions

    def in_bounds(self,x,y):
        return x >= 0 and x <= 7 and y >= 0 and y <= 7

    ddef ray_flip(self, origin_x, origin_y, target_x, target_y):
        #**ADD INBOUNDS CHECK
        new_state = self.clone()
        new_state.set(origin_x, origin_y, self.friendly())
        dx = target_x - origin_x
        dy = target_y - origin_y
        piece = self.at(target_x, target_y)
        while piece != self.friendly():
            if piece == self.enemy(): # flip piece
                new_state.set(target_x, target_y, self.friendly())
            elif piece == 0: # no end piece
                raise ValueError('No bracketing piece.')
            target_x += dx
            target_y += dy
            piece = self.at(target_x, target_y)
        return new_state

    def isomorphisms(self):
        reflections = [self, self.reflect('x'), self.reflect('y'), self.reflect('xy')]
        rotations = []
        for reflection in reflections:
            rotations.append(reflection.rotate_clockwise())
        return reflections + rotations

    def reflect(self, axis):
        reflection = State()
        for y in range(8):
            for x in range(8):
                if   axis ==  'x': reflection.set(7-x,   y, self.at(x,y))
                elif axis ==  'y': reflection.set(x,   7-y, self.at(x,y))
                elif axis == 'xy': reflection.set(7-x, 7-y, self.at(x,y))
        return reflection

    def rotate_clockwise(self):
        rotation = State()
        for y in range(8):
            for x in range(8):
                rotation.set(7-y, x, self.at(x,y))
        return rotation

    def pretty_print(self):
        h_border = ' | | | | | | | | | |'
        v_border = '|'
        print('\n   0 1 2 3 4 5 6 7     ',
            self.player_symbol(self.to_move), ' to move')
        print(h_border)
        for y in range(8):
            print('', v_border, end=' ')
            for x in range(8):
                p = self.at(x,y)
                print(self.player_symbol(p), end=' ')
            print(v_border, y, sep=' ')
        print(h_border);
        print(self.score())

    def player_symbol(self,n):
        return { 0:'.', 1.:'x', 2.:'o' }[int(n)]

    def score(self):
        score = [0, 0]
        for y in range(8):
            for x in range(8):
                val = self.at(x, y)
                if val == 1:
                    score[0] += 1
                elif val == 2:
                    score[1] += 1
        return score

def main():
    state = State()
    print(state.board[0])
    print('Starting State:')
    print(state.board)
    print("\n\n")
    state.pretty_print()
    print('Legal Moves:')
    for move_state in state.legal_moves():
        move_state.pretty_print()



if  __name__ =='__main__':main()
