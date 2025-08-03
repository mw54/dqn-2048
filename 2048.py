import numpy as np

class Board:
    def __init__(self, size:int=4, seed:int=42):
        self.action_map = {
            "left": (0, 0),
            "right": (0, 1),
            "up": (1, 0),
            "down": (1, 1)
        }
        np.random.seed(seed)
        self.board = np.zeros((size, size), int)
        self.actions = list()
        self.reset()

    def reset(self):
        self.game_over = False
        self.score = 0
        self.board.fill(0)
        self._add_tile()
        self._add_tile()
        self._update_actions()

    def _add_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if len(empty_cells) > 0:
            cell_index = np.random.choice(len(empty_cells))
            row_index, col_index = empty_cells[cell_index]
            self.board[row_index, col_index] = 2 if np.random.random() < 0.9 else 4
    
    def _transpose_board(self):
        self.board = self.board.transpose()

    def _flip_board(self):
        self.board = self.board[:,::-1]
    
    def _update_board(self, action:str) -> int:
        reward = 0
        if self.action_map[action][0]:
            self._transpose_board()
        if self.action_map[action][1]:
            self._flip_board()
        for row in self.board:
            tiles = row[row != 0]
            merged = list()
            index = 0
            while index <= len(tiles) - 1:
                if index == len(tiles) - 1:
                    merged.append(tiles[index])
                    index += 1
                elif tiles[index] == tiles[index + 1]:
                    merged.append(tiles[index] + tiles[index + 1])
                    reward += tiles[index] + tiles[index + 1]
                    index += 2
                else:
                    merged.append(tiles[index])
                    index += 1
            row[row != 0] = 0
            for index, value in enumerate(merged):
                row[index] = value
        if self.action_map[action][1]:
            self._flip_board()
        if self.action_map[action][0]:
            self._transpose_board()
        return reward

    def _update_actions(self) -> list[str]:
        self.actions.clear()
        self.backup = self.board.copy()
        for action in self.action_map:
            self._update_board(action)
            if not np.array_equal(self.board, self.backup):
                self.actions.append(action)
            self.board = self.backup.copy()
    
    def step(self, action:str) -> int:
        if self.game_over:
            raise ValueError("game is in final state")
        if action not in self.actions:
            raise ValueError(f"invalid action: {action}")
        reward = self._update_board(action)
        self._add_tile()
        self.score += reward
        self._update_actions()
        if len(self.actions) == 0:
            self.game_over = True
        return reward
    
    def __repr__(self) -> str:
        output = f"Score: {self.score}\n"
        output += ("+" + "-" * 4) * 4 + "+\n"
        
        for row in self.board:
            output += "|"
            for cell in row:
                if cell == 0:
                    output += "    |"
                else:
                    output += f"{cell:4d}|"
            output += "\n"
            output += ("+" + "-" * 4) * 4 + "+\n"
        
        if self.game_over:
            output += "GAME OVER!"
        else:
            output += f"Valid actions: {self.actions}"
        
        return output
    
if __name__ == "__main__":
    # Test the environment
    env = Board()
    print(env)
    
    # Play a few moves
    while not env.game_over:
        action = input("Action: ")
        reward = env.step(action)
        print(f"Reward: {reward}")
        print(env)
