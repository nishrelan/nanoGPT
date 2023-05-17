import pyspiel as ps
from data.checkers.prepare import rows, cols, chartoint, inttochar


# TODO: Implement for when using annotations
class CheckersBoard:

    def __init__(self, move_list=None, annotations=None):
        if move_list is None:
            self.game = ps.load_game('checkers')
            self.state = self.game.new_initial_state()
        else:
            self.game, self.state = self._init_from_game_string(move_list, annotations)
    
    def _transform(self, move):
        return move[0] + str(9 - int(move[1])) + move[2] + str(9 - int(move[3]))

    def _init_from_game_string(self, move_list, annotations=False):
        """
        game string is of form ['a3b4', 'c3d5', ...] with characters 'x' and '|' only if
        annotations is True
        """
        move_list = [self._transform(m) for m in move_list]
        # do without annotations first
        game = ps.load_game('checkers')
        state = game.new_initial_state()
        for move in move_list:
            las = state.legal_actions()
            move_strings = [state.action_to_string(state.current_player(), la) for la in las]
            idx = move_strings.index(move) # ValueError indicates ilegal move
            state.apply_action(las[idx])
        
        return game, state

    def get_legal_moves(self, annotations=False):
        """
        Returns list legal moves in standard form, add annotations if annotations is True 
        """
        las = self.state.legal_actions()
        return [self._transform(self.state.action_to_string(self.state.current_player(), la)) for la in las]

    def is_legal(self, move):
        return move in self.get_legal_moves()
       



    def make_move(self, move, annotations=False):
        move = self._transform(move)
        las = self.state.legal_actions()
        move_strings = [self.state.action_to_string(self.state.current_player(), la) for la in las]
        idx = move_strings.index(move) # ValueError indicates ilegal move
        self.state.apply_action(las[idx])





# game = ps.load_game('checkers')
# state = game.new_initial_state()
# print(state)
# las = state.legal_actions()
# legal_moves = [state.action_to_string(state.current_player(), la) for la in las]
# move = legal_moves[0]
# print(legal_moves)
# state.apply_action(las[0])
# print(state)


