import os

def print_tree(startpath, prefix=""):
    """Recursively prints the folder structure."""
    items = sorted(os.listdir(startpath))
    for i, item in enumerate(items):
        path = os.path.join(startpath, item)
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + item)
        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    root_dir = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/fr3_controllers"
    print(root_dir)
    print_tree(root_dir)
