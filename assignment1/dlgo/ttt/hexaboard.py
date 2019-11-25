import copy

from dlgo.ttt.ttttypes import Player, Point

__all__ = [
    'Board',
    'GameState',
    'Move',
]


class IllegalMoveError(Exception):
    pass


BOARD_SIZE = 3
ROWS = tuple(range(1, BOARD_SIZE + 1))
COLS = tuple(range(1, BOARD_SIZE + 1))
# Top left to lower right diagonal
#DIAG_1 = (Point(1, 1), Point(2, 2))

# Top right to lower left diagonal
#DIAG_2 = (Point(1, 3), Point(2, 2), Point(3, 1))


class Board:
    def __init__(self):
        self._grid = {Point(row=1,col=1):Player.x, Point(row=1,col=2):Player.x,Point(row=1,col=3):Player.x,
        Point(row=3,col=1):Player.o, Point(row=3,col=2):Player.o,Point(row=3,col=3):Player.o}

    def place(self, player, point):
        do_place = False
        # Check if point is on grid
        assert self.is_on_grid(point)
        # ttt check if position is empty
        #assert self._grid.get(point) is None

        
        #Check if point a diagonal containing the opponent
        if player == Player.x:
            print("x>1", point.row,point.col, self._grid.get(point.row,point.col), self._grid.get(point.row - 1,point.col - 1),self._grid.get(point.row - 1 ,point.col + 1) )

            if self._grid.get(point.row,point.col) == Player.o \
                and (
                    self._grid.get(point.row - 1,point.col - 1) == Player.x \
                    or \
                    self._grid.get(point.row - 1 ,point.col + 1) == Player.x \
                ):
                do_place = True

        #Check if point is empty or it is a diagonal containing the opponent
        if player == Player.o:
            if self._grid.get(point.row,point.col) == Player.x \
                and (
                    self._grid.get(point.row + 1,point.col - 1) == Player.o \
                    or \
                    self._grid.get(point.row + 1 ,point.col + 1) == Player.o \
                ):
                do_place = True
                
        #Moving to an empty space (straigh line, with check)
        if self._grid.get(point) is None:
            #Erase former piece
            if player == Player.x and self._grid[Point(point.row-1,point.col)] == Player.x:
                self._grid[Point(point.row-1,point.col)] = None
                do_place = True
            if player == Player.o and self._grid[Point(point.row+1,point.col)] == Player.o:
                self._grid[Point(point.row+1,point.col)] = None
                do_place = True
                    
        # Place the pawn
        if do_place:
            self._grid[point] = player
            
        # Erase the former pawn
        
        

    @staticmethod
    def is_on_grid(point):
        return 1 <= point.row <= BOARD_SIZE and \
            1 <= point.col <= BOARD_SIZE

    def get(self, point):
        """Return the content of a point on the board.
        Returns None if the point is empty, or a Player if there is a
        stone on that point.
        """
        #print(self._grid)
        return self._grid.get(point)


class Move:
    def __init__(self, point):
        self.point = point  #Returns Point(row=1, col=2)


class GameState:
    def __init__(self, board, next_player, move):
        self.board = board
        self.next_player = next_player
        self.last_move = move

    def apply_move(self, move):
        """Return the new GameState after applying the move."""
        next_board = copy.deepcopy(self.board)
        next_board.place(self.next_player, move.point)
        return GameState(next_board, self.next_player.other, move)

    @classmethod
    def new_game(cls):
        print("HexaPawn Mode")
        board = Board()
        return GameState(board, Player.x, None)

    def is_valid_move(self, move):
        is_valid = False
        #print(move.point)
        #Check if point is empty or it is a diagonal containing the opponent
        #if self.next_player == Player.x:
        #    player = Player.o
        #else:
        #    player = Player.x
        player = self.next_player
            
        #X never can play in its initial line (never gets back)
        if move.point.row <= 1 and player == Player.x:
            return False
        #O never can play in its initial line (never gets back)
        if move.point.row >= 3 and player == Player.o:
            return False
        
        if player == Player.x and move.point.row > 1:
            if self.board._grid.get(move.point.row,move.point.col) == Player.o \
                and (
                    self.board._grid.get(move.point.row - 1,move.point.col - 1) == Player.x \
                    or \
                    self.board._grid.get(move.point.row - 1 ,move.point.col + 1) == Player.x \
                ):
                is_valid = True

        #Check if point is empty or it is a diagonal containing the opponent
        if player == Player.o and move.point.row < 3:
            if self.board._grid.get(move.point.row,move.point.col) == Player.x \
                and (
                    self.board._grid.get(move.point.row + 1,move.point.col - 1) == Player.o \
                    or \
                    self.board._grid.get(move.point.row + 1 ,move.point.col + 1) == Player.o \
                ):
                is_valid = True

        #Moving to an empty space (straigh line, with check)
        if self.board._grid.get(move.point) is None: 
            #print("o empty", player, move.point.row, "  ", self.board._grid[Point(move.point.row+1,move.point.col)])
            #Check if the piece was at its previous position
            if player == Player.x and move.point.row > 1 \
                and self.board._grid[Point(move.point.row-1,move.point.col)] == Player.x:
                is_valid = True
            if player == Player.o  and move.point.row < 3 \
                and self.board._grid[Point(move.point.row+1,move.point.col)] == Player.o:
                is_valid = True


        #print(is_valid)
            
        return is_valid
        
    def legal_moves(self):
        moves = []
        for row in ROWS:
            for col in COLS:
                move = Move(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        return moves

    def is_over(self):

        # ttt implementation commented
        #if self._has_3_in_a_row(Player.x):
        #    return True
        #if self._has_3_in_a_row(Player.o):
        #    return True
        #if all(self.board.get(Point(row, col)) is not None
        #       for row in ROWS
        #       for col in COLS):
        #    return True

        # Hexapawn Test for both players
        if self.player_reached_opposite_side(Player.x):
            return True
        if self.player_reached_opposite_side(Player.o):
            return True

        # Hexapawn always has no more valid moves.
        if len(self.legal_moves()) == 0:
            return True

        return False
        
    # Hexapawn implementation of check a winner    
    def player_reached_opposite_side(self, player):
        # If x reached row 3 - x wins; if o reached row 1 - o wins
        if player == player.x:
            row = 3
        else: 
            row = 1
        # Check if any piece has reached the opposite side
        for col in COLS:
            if self.board.get(Point(row,col)) == player:
                return True
        return False
        

    def _has_3_in_a_row(self, player):
        # Vertical
        for col in COLS:
            if all(self.board.get(Point(row, col)) == player for row in ROWS):
                return True
        # Horizontal
        for row in ROWS:
            if all(self.board.get(Point(row, col)) == player for col in COLS):
                return True
        # Diagonal UL to LR
        if self.board.get(Point(1, 1)) == player and \
                self.board.get(Point(2, 2)) == player and \
                self.board.get(Point(3, 3)) == player:
            return True
        # Diagonal UR to LL
        if self.board.get(Point(1, 3)) == player and \
                self.board.get(Point(2, 2)) == player and \
                self.board.get(Point(3, 1)) == player:
            return True
        return False

    def winner(self):
        
        # Hexapawn - Test for both players
        if self.player_reached_opposite_side(Player.x):
            return Player.x
        if self.player_reached_opposite_side(Player.o):
            return Player.o
         
        # Old ttt implementation    
        #if self._has_3_in_a_row(Player.x):
        #    return Player.x
        #if self._has_3_in_a_row(Player.o):
        #    return Player.o
        return None