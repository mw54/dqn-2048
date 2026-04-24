import time

class Terminal:
    def __enter__(self):
        print("\x1b[?1049h\x1b[H", end="")
        return self

    def sleep(self, seconds:float|bool=False):
        if type(seconds) is float:
            time.sleep(seconds)
        elif type(seconds) is bool and seconds:
            input()
        else:
            raise ValueError(f"invalid type for sleep: {type(seconds)}")

    def clear(self):
        print("\x1b[2J\x1b[H", end="")
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("\x1b[?1049l", end="")
