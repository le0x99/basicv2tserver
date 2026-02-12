import os
from server import ui_server

if __name__ == "__main__":
    # Ensure UI does NOT use GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ui_server.launch()
