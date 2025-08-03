import torch

DEVICE = torch.device("cpu")
torch.set_grad_enabled(False)

def action_encode(text:list[str]) -> torch.Tensor:
    mapping = {"left": [0, 0], "right": [0, 1], "up": [1, 0], "down": [1, 1]}
    encoded = list()
    for word in text:
        encoded.append(mapping[word])
    return torch.tensor(encoded, dtype=torch.bool, device=DEVICE)

def action_decode(actions:torch.Tensor) -> list[str]:
    assert actions.shape == (4,)
    mapping = {0: "left", 1: "right", 2: "up", 3: "down"}
    indices = torch.nonzero(actions).flatten().tolist()
    decoded = list()
    for index in indices:
        decoded.append(mapping[index])
    return decoded

class BatchBoards:
    def __init__(self, board_size:int=4, batch_size:int=3):
        self.board_size = board_size
        self.batch_size = batch_size
        self.boards = torch.zeros((batch_size, board_size, board_size), dtype=torch.int, device=DEVICE)
        self.scores = torch.zeros(batch_size, dtype=torch.int, device=DEVICE)
        self.terminals = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
        self.actions = torch.zeros((batch_size, 4), dtype=torch.bool, device=DEVICE)
        self.reset()

    def reset(self):
        self.boards[self.terminals] = 0
        self.scores[self.terminals] = 0
        self._add_tiles(reset=True)
        self._add_tiles(reset=True)
        self.terminals.zero_()
        self._update_actions()

    def _add_tiles(self, reset:bool=False):
        self.boards = self.boards.view(self.batch_size, -1)
        available = torch.eq(self.boards, 0)
        if reset:
            available[~self.terminals] = 0
        
        indices = (torch.rand(self.batch_size, device=DEVICE) * torch.sum(available, dim=1, dtype=torch.float)).to(torch.int)
        indices = indices.unsqueeze(1) + 1
        
        mask = (torch.cumsum(available, dim=1, dtype=torch.int) == indices) & available
        values = torch.where(torch.rand(self.batch_size, device=DEVICE) < 0.9, 2, 4).to(torch.int)

        tile_matrix = values.unsqueeze(1).expand(-1, self.board_size * self.board_size)
        self.boards.masked_scatter_(mask, tile_matrix[mask])
        self.boards = self.boards.view(self.batch_size, self.board_size, self.board_size)
    
    def _merge_tiles(self) -> torch.Tensor:
        equal = torch.eq(self.boards[:,:,:-1], self.boards[:,:,1:])
        for i in range(self.board_size - 2):
            equal[:,:,i + 1] = (~equal[:,:,i]) & equal[:,:,i + 1]
        self.boards[:,:,:-1][equal] += self.boards[:,:,1:][equal]
        self.boards[:,:,1:][equal] = 0
        rewards = torch.sum(self.boards[:, :, :-1] * equal, dim=(1, 2), dtype=torch.int)
        return rewards

    def _push_tiles(self):
        mask = torch.ne(self.boards, 0)
        tile_counts = torch.sum(mask, dim=2, keepdim=True, dtype=torch.int)
        range_index = torch.arange(self.board_size, device=DEVICE)[None, None, :]
        indices = torch.lt(range_index, tile_counts)
        self.boards[indices] = self.boards[mask]
        self.boards[(~indices) & mask] = 0

    def _update_boards(self, actions:torch.Tensor) -> torch.Tensor:
        trans_mask = actions[:,0]
        flip_mask = actions[:,1]
        self.boards[trans_mask] = torch.transpose(self.boards[trans_mask], 1, 2)
        self.boards[flip_mask] = torch.flip(self.boards[flip_mask], dims=[2])
        self._push_tiles()
        rewards = self._merge_tiles()
        self._push_tiles()
        self.boards[flip_mask] = torch.flip(self.boards[flip_mask], dims=[2])
        self.boards[trans_mask] = torch.transpose(self.boards[trans_mask], 1, 2)
        return rewards

    def _check_actions(self, actions:torch.Tensor) -> torch.Tensor:
        backup = torch.clone(self.boards)
        self._update_boards(actions)
        admissive = torch.any(torch.ne(backup, self.boards), dim=(1, 2))
        self.boards = backup
        return admissive
        
    def _update_actions(self):
        self.actions.zero_()
        actions = torch.zeros((self.batch_size, 2), dtype=torch.bool, device=DEVICE)
        for i in range(4):
            actions.zero_()
            actions[:,0] = i // 2
            actions[:,1] = i % 2
            self.actions[:,i] = self._check_actions(actions)
        self.terminals = ~torch.any(self.actions, dim=1)
        
    def __call__(self, actions:torch.Tensor) -> torch.Tensor:
        if actions.shape != (self.batch_size, 2) or actions.dtype != torch.bool:
            raise ValueError("incorrect action tensor format")
        if not torch.all(self.actions[torch.arange(self.batch_size, device=DEVICE),actions[:,0] * 2 + actions[:,1]][~self.terminals]):
            raise ValueError("action unavailable")
        rewards = self._update_boards(actions)
        self._add_tiles()
        self.scores += rewards
        self._update_actions()
        return rewards
    
    def __repr__(self) -> str:
        outputs = ""
        for i in range(self.batch_size):
            outputs += "\n"
            outputs += f"Board: {i}, Score: {self.scores[i]}\n"
            outputs += ("+" + "-" * 4) * self.board_size + "+\n"
            for j in range(self.board_size):
                outputs += "|"
                for k in range(self.board_size):
                    cell = self.boards[i,j,k]
                    if cell == 0:
                        outputs += "    |"
                    else:
                        outputs += f"{cell:4d}|"
                outputs += "\n"
                outputs += ("+" + "-" * 4) * self.board_size + "+\n"
            if self.terminals[i]:
                outputs += "GAME OVER!\n"
            else:
                outputs += f"Available: {action_decode(self.actions[i])}\n"
        return outputs

if __name__ == "__main__":
    # Test the environment
    env = BatchBoards()
    print(env)
    
    # Play a few moves
    while not torch.all(env.terminals):
        actions = input("Actions: ")
        actions = action_encode(actions.split())
        rewards = env(actions)
        print(f"Rewards: {rewards.tolist()}")
        print(env)

