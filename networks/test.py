import difflib


with open('miniworld.py') as file_1:
    file_1_text = file_1.readlines()

with open('jason_miniworld.py') as file_2:
    file_2_text = file_2.readlines()


# Find and print the diff:
for line in difflib.unified_diff(
        file_1_text, file_2_text, fromfile='file1.txt', 
        tofile='file2.txt', lineterm=''):
    print(line)

