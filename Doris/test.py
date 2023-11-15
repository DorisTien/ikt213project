import os

folder_path = r'C:\Users\doris\Downloads\Assets\Training\mypic\nonplastic'

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Create a list of full file paths
file_paths = [os.path.join(folder_path, file_name) for file_name in file_list]

# Print and store the list of file paths
print("File paths in the folder:")
i=0
for file_path in file_paths:
    i=1+i
    print(i)
    print('"'+file_path+'",')

# If you want to store the paths in a variable for later use:
# Assuming you want to use this list somewhere else in your code
# (outside of this example)
# You can then use this 'file_paths' list elsewhere in your program.

