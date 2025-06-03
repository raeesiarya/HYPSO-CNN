import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libraries import *

#######################################################################################
#######################################################################################
#######################################################################################

"""
The functions move_forward(), move_back() and list_contents() are used for retrieving
.dat, .bip (/.bip@) and .png files for use in CSV creation.
"""

current_path = Path(".").resolve()

def move_forward(folder_name):
    global current_path
    new_path = current_path / folder_name
    if new_path.exists() and new_path.is_dir():
        current_path = new_path
        return f"Moved to: {current_path}"
    return "Folder does not exist."

def move_back():
    global current_path
    if current_path.parent.exists():
        current_path = current_path.parent
        return f"Moved back to: {current_path}"
    return "Cannot move back further."

def list_contents():
    return [item.name for item in current_path.iterdir()]

#######################################################################################

def get_dat_png_bip():
    """
    Collects and returns paths of .dat, .png, and .bip files from specified directories.
    This function navigates through a directory structure to find and collect paths of files
    with specific extensions (.dat, .png, .bip) from labeled and raw data directories.
    Returns:
        tuple: A tuple containing three lists:
            - dat_paths (list): List of paths to .dat files.
            - png_paths (list): List of paths to .png files.
            - bip_paths (list): List of paths to .bip files.
    """

    dat_paths = []
    dat_paths_checkup = []

    png_paths = []
    bip_paths = []

    move_forward("corrected_labeled_data_CLOUD")
    contents_labeled_data = list_contents()
    for j in contents_labeled_data:
        dat_paths.append((str(current_path / j))[35:])
        dat_paths_checkup.append(j[:j.find("-l1a")])
        #print(j[:j.find("-l1a")])

    move_back()
    
    #print(os.getcwd())


    dat_paths_checkup = ([str(path) for path in dat_paths_checkup])
    dat_paths_checkup = sorted(dat_paths_checkup)
    #print(len(dat_paths_checkup))
    #print(dat_paths_checkup)

    move_forward("raw_data")
    L1_contents = list_contents()
    #print(L1_contents)

    print(dat_paths_checkup)

    for folder in L1_contents:
        move_forward(folder)
        L2_contents = list_contents()
        print(colored(L2_contents, "blue"))
        for folder_gr_2 in L2_contents:
            if folder_gr_2 in dat_paths_checkup:
                move_forward(folder_gr_2)
                L3_contents = list_contents()
                print(colored(L3_contents, "light_blue"))
                for i in L3_contents:
                    if i.endswith("Z.png"):
                        png_paths.append((str(current_path / i))[35:])
                    elif i.endswith(".bip"):
                        bip_paths.append((str(current_path / i))[35:])
                    elif i.endswith(".bip@"):
                        bip_paths.append((str(current_path / i))[35:])

                move_back()
        move_back()
    move_back()

    print("-"*100)
    print(sorted(png_paths))

    print(len(dat_paths), len(png_paths), len(bip_paths))

    return dat_paths, png_paths, bip_paths

#######################################################################################

def create_csv_file():
    """
    Creates CSV files that maps .dat files 
    to their corresponding .bip (/.bip@) and .png files.

    - Calls get_dat_png_bip() to retrieve paths.
    - Writes the file paths into 'train_files.csv' and 
    'evaluate_files.csv' with headers: ["dat_files", "bip_files", "png_files"].
    """

    os.makedirs('csv', exist_ok=True)

    dat_files_paths, png_files_paths, bip_files_paths = get_dat_png_bip()

    dat_files_paths = sorted(dat_files_paths)
    png_files_paths = sorted(png_files_paths)
    bip_files_paths = sorted(bip_files_paths)

    if len(dat_files_paths) == 1:
        with open('csv/train_files.csv', mode='w') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(["dat_files", "bip_files", "png_files"])
            writer.writerow([dat_files_paths[0], bip_files_paths[0], png_files_paths[0]])
    elif len(dat_files_paths) > 1:
        split_index = int(len(dat_files_paths)*0.9)
        train_dat_files = dat_files_paths[:split_index]
        eval_dat_files = dat_files_paths[split_index:]

        train_bip_files = bip_files_paths[:split_index]
        eval_bip_files = bip_files_paths[split_index:]

        train_png_files = png_files_paths[:split_index]
        eval_png_files = png_files_paths[split_index:]

        with open('csv/train_files.csv', mode='w') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(["dat_files", "bip_files", "png_files"])
            for i in range(len(train_dat_files)):
                writer.writerow([train_dat_files[i], train_bip_files[i], train_png_files[i]])

        with open('csv/evaluate_files.csv', mode='w') as eval_file:
            writer = csv.writer(eval_file)
            writer.writerow(["dat_files", "bip_files", "png_files"])
            for i in range(len(eval_dat_files)):
                writer.writerow([eval_dat_files[i], eval_bip_files[i], eval_png_files[i]])
    else:
        with open('csv/train_files.csv', mode='w') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(["dat_files", "bip_files", "png_files"])
            for i in range(len(dat_files_paths)):
                writer.writerow([dat_files_paths[i], bip_files_paths[i], eval_png_files[i]])

#######################################################################################

def read_csv_file(csv_file_with_path):
    """
    Reads a CSV file containing .dat and .bip@ file paths.

    Parameters:
    - csv_file_with_path (str): Path to the CSV file.

    Returns:
    - bip_files (list): List of .bip (/.bip@) file paths.
    - dat_files (list): List of .dat file paths.
    - png_files (list): List of .png file paths.
    """
    
    dat_files = []
    bip_files = []
    png_files = []

    with open(f'{csv_file_with_path}', mode='r') as file:
        csvFile = csv.reader(file)

        i = 0
        for lines in csvFile:
            if i != 0:
                dat_files.append(lines[0])
                bip_files.append(lines[1])
                png_files.append(lines[2])
            i += 1
    
    return bip_files, dat_files, png_files

#######################################################################################

#create_csv_file()