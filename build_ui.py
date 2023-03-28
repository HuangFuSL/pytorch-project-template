import os


def compile():
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.ui'):
                i = os.path.join(root, file)
                o = os.path.join(root, file.replace('.ui', '_ui.py'))
                os.system(f"pyside6-uic {i} -o {o}")
