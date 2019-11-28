import copy

from dlgoo.ttt.ttttypes import Player, Point

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
        #Populate the Hexapawn board
        self._grid = {Point(row=1,col=1):Player.x, Point(row=1,col=2):Player.x,Point(row=1,col=3):Player.x,
        Point(row=3,col=1):Player.o, Point(row=3,col=2):Player.o,Point(row=3,col=3):Player.o}

    def place(self, player, point):
        do_place = False
        # Check if point is on grid
        assert self.is_on_grid(point)
        #Moving to an empty space (straigh line, with check)
        if self._grid.get(point) is None:
            #Erase former piece
            if player == Player.x and self._grid[Point(point.row-1,point.col)] == Player.x:
                # Erase the former pawn
                self._grid[Point(point.row-1,point.col)] = None
                # Place the pawn
                self._grid[point] = player
                return
            if player == Player.o and self._grid[Point(point.row+1,point.col)] == Player.o:
                # Erase the former pawn                
                self._grid[Point(point.row+1,point.col)] = None
                # Place the pawn
                self._grid[point] = player
                return

        
        #Check if point a diagonal containing the opponent
        if player == Player.x and do_place == False:
            """
            print("Place X", point.row,point.col, self._grid.get(Point(point.row,point.col)),\
                 self._grid.get(Point(point.row - 1,point.col - 1 )), \
                 self._grid.get(Point(point.row - 1 ,point.col + 1 )) )
            """
            if self._grid.get(Point(point.row,point.col)) == Player.o \
              and self._grid.get(Point(point.row - 1,point.col - 1)) == Player.x:
                # Erase the former pawn
                self._grid[Point(point.row-1,point.col-1)] = None
                # Place the pawn
                self._grid[point] = player
                return

            if self._grid.get(Point(point.row,point.col)) == Player.o \
              and self._grid.get(Point(point.row - 1 ,point.col + 1)) == Player.x:
                # Erase the former pawn
                self._grid[Point(point.row-1,point.col+1)] = None            
                # Place the pawn
                self._grid[point] = player
                return

        #Check if point is empty or it is a diagonal containing the opponent
        if player == Player.o and do_place == False:
            if self._grid.get(Point(point.row,point.col)) == Player.x \
              and self._grid.get(Point(point.row + 1,point.col - 1)) == Player.o:
                # Erase the former pawn
                self._grid[Point(point.row+1,point.col-1)] = None   
                # Place the pawn
                self._grid[point] = player
                return
            
            if self._grid.get(Point(point.row,point.col)) == Player.x \
              and self._grid.get(Point(point.row + 1 ,point.col + 1)) == Player.o:
                # Erase the former pawn
                self._grid[Point(point.row+1,point.col+1)] = None   
                # Place the pawn
                self._grid[point] = player
                return
                    
        # Place the pawn
        #if do_place:
        #    self._grid[point] = player
            
 
        
        

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
        
        #Check if point is has opponent or it is a diagonal containing the opponent
        if player == Player.x and move.point.row > 1:
            #Check if the targeted poin has the opponent
            if self.board._grid.get(move.point) == Player.o \
                and (
                    # check if the previous position is a diagonal and contains the x player. any of them may be true
                    self.board._grid.get(Point(move.point.row - 1,move.point.col - 1)) == Player.x \
                    or \
                    self.board._grid.get(Point(move.point.row - 1 ,move.point.col + 1)) == Player.x \
            ):
                """
                print("isv_X>1", move.point.row,move.point.col, self.board._grid.get(Point(move.point.row,move.point.col)),\
                        self.board._grid.get(Point(move.point.row - 1,move.point.col - 1 )), \
                        self.board._grid.get(Point(move.point.row - 1,move.point.col + 1 )) )
                """       
                is_valid = True

        if player == Player.o and move.point.row < 3:
            if self.board._grid.get(move.point) == Player.x \
                and (
                    self.board._grid.get(Point(move.point.row + 1,move.point.col - 1)) == Player.o \
                    or \
                    self.board._grid.get(Point(move.point.row + 1 ,move.point.col + 1)) == Player.o \
            ):
                """    
                print("isv_Y>1", move.point.row,move.point.col, self.board._grid.get(Point(move.point.row,move.point.col)),\
                        self.board._grid.get(Point(move.point.row + 1,move.point.col - 1 )), \
                        self.board._grid.get(Point(move.point.row + 1,move.point.col + 1 )) )
                """            
                is_valid = True
        #Moving to an empty space (straigh line, with check)
        if self.board._grid.get(move.point) is None: 
            #print("o empty", player, move.point.row, "  ", self.board._grid[Point(move.point.row+1,move.point.col)])
            #Check if the piece was at its previous position
            if player == Player.x and move.point.row > 1 \
                and self.board._grid.get(Point(move.point.row-1,move.point.col)) == Player.x:
                is_valid = True
            if player == Player.o  and move.point.row < 3 \
                and self.board._grid.get(Point(move.point.row+1,move.point.col)) == Player.o:
                is_valid = True

            
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

        # Hexapawn Test for both players
        if self.player_reached_opposite_side(Player.x):
            return True
        if self.player_reached_opposite_side(Player.o):
            return True

        # Hexapawn - Check if it has no more valid moves.
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
        
    #Former method for ttt
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
            
        # If no more moves, the last player that played is the winner    
        if self.next_player == Player.x:    
            return Player.o
        else:
            return Player.x
            
         
        # Old ttt implementation    
        #if self._has_3_in_a_row(Player.x):
        #    return Player.x
        #if self._has_3_in_a_row(Player.o):
        #    return Player.o
        #return None