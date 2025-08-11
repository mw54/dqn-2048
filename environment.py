import torch

DEVICE = torch.device("cpu")
torch.set_grad_enabled(False)

def action_encode(text:list[str]) -> torch.Tensor:
    mapping = {"left": [0, 0], "right": [0, 1], "up": [1, 0], "down": [1, 1]}
    encoded = list()
    for word in text:
        encoded.append(mapping[word])
    return torch.tensor(encoded, dtype=torch.bool, device=DEVICE)

class BatchBoards:
    def __init__(self, board_size:int=4, batch_size:int=3):
        self.board_size = board_size
        self.batch_size = batch_size
        self.boards = torch.zeros((batch_size, board_size, board_size), dtype=torch.int, device=DEVICE)
        self.scores = torch.zeros(batch_size, dtype=torch.int, device=DEVICE)
        self.terminals = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
        self.reset()

    def reset(self):
        self.boards[self.terminals] = 0
        self.scores[self.terminals] = 0
        self._add_tiles(reset=True)
        self._add_tiles(reset=True)
        self.terminals.zero_()

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
        backup = torch.clone(self.boards)
        trans_mask = actions[:,0]
        flip_mask = actions[:,1]
        self.boards[trans_mask] = torch.transpose(self.boards[trans_mask], 1, 2)
        self.boards[flip_mask] = torch.flip(self.boards[flip_mask], dims=[2])
        self._push_tiles()
        rewards = self._merge_tiles()
        self._push_tiles()
        self.boards[flip_mask] = torch.flip(self.boards[flip_mask], dims=[2])
        self.boards[trans_mask] = torch.transpose(self.boards[trans_mask], 1, 2)
        self.terminals = torch.all(torch.eq(backup, self.boards), dim=(1, 2))
        return rewards
                
    def __call__(self, actions:torch.Tensor) -> torch.Tensor:
        if actions.shape != (self.batch_size, 2) or actions.dtype != torch.bool:
            raise ValueError("incorrect action tensor format")
        rewards = self._update_boards(actions)
        self._add_tiles()
        self.scores += rewards
        return rewards
    
    def __repr__(self) -> str:
        width = max(4, int(torch.log10(self.boards.max())) + 1)
        outputs = ""
        for i in range(self.batch_size):
            outputs += "\n"
            outputs += f"Board: {i}, Score: {self.scores[i]}\n"
            outputs += ("+" + "-" * width) * self.board_size + "+\n"
            for j in range(self.board_size):
                outputs += "|"
                for k in range(self.board_size):
                    cell = self.boards[i,j,k]
                    if cell == 0:
                        outputs += " " * width + "|"
                    else:
                        outputs += f"{cell:{width}d}" + "|"
                outputs += "\n"
                outputs += ("+" + "-" * width) * self.board_size + "+\n"
            if self.terminals[i]:
                outputs += "GAME OVER!\n"
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

