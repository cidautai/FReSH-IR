import os
import random

def create_path(IMGS_PATH, list_new_files):
    '''
    Util function to add the file path of all the images to the list of names of the selected 
    images that will form the valid ones.
    '''
    file_path, name = os.path.split(
        IMGS_PATH[0])  # we pick only one element of the list
    output = [os.path.join(file_path, element) for element in list_new_files]

    return output

def common_member(a, b):
    """
    Determine if two lists have at least one common element.

    This function checks whether there is any overlap between two input lists
    by converting them to sets and finding their intersection.

    Args:
        a (list): The first list to compare.
        b (list): The second list to compare.

    Returns:
        bool: True if the two lists have at least one common element, False otherwise.
    """
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False


def random_sort_pairs(list1, list2):
    """
    Randomly shuffles two lists while maintaining the pairing between their elements.

    This function takes two lists of equal length, combines them into pairs, 
    shuffles the pairs randomly, and then separates them back into two lists. 
    The pairing between elements in the two lists is preserved after the shuffle.

    Args:
        list1 (list): The first list to be shuffled.
        list2 (list): The second list to be shuffled. Must have the same length as list1.

    Returns:
        tuple: A tuple containing the shuffled versions of list1 and list2, 
               with their original pairing maintained.

    Raises:
        ValueError: If the input lists are not of the same length.

    Example:
        list1 = [1, 2, 3]
        list2 = ['a', 'b', 'c']
        shuffled_list1, shuffled_list2 = random_sort_pairs(list1, list2)
        # shuffled_list1 and shuffled_list2 will be shuffled but still paired, e.g.,
        # shuffled_list1 = [3, 1, 2], shuffled_list2 = ['c', 'a', 'b']
    """
    # Combine the lists
    combined = list(zip(list1, list2))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip back into separate lists
    list1[:], list2[:] = zip(*combined)

    return list1, list2

def flatten_list_comprehension(matrix):
    """
    Flattens a 2D list (list of lists) into a 1D list using list comprehension.

    Args:
        matrix (list of list): A 2D list where each element is a list.

    Returns:
        list: A flattened 1D list containing all elements from the 2D list.
    """
    return [item for row in matrix for item in row]

def separate_elements(file_path):
    """
    Reads a file and separates its content into two lists based on line structure.

    This function processes a file where each line is expected to contain two 
    space-separated elements. It splits each line into two parts and appends 
    the first part to one list and the second part to another list.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        tuple: A tuple containing two lists:
            - The first list contains the first elements from each line.
            - The second list contains the second elements from each line.

    Note:
        Lines that do not contain exactly two space-separated elements are ignored.
    """
    list1 = []
    list2 = []

    with open(file_path, 'r') as file:
        for line in file:
            elements = line.strip().split()
            if len(elements) == 2:
                list1.append(elements[0])
                list2.append(elements[1])

    return list1, list2

def read_file(file_path):
    """
    Reads a file line by line, extracts the first element from each line after splitting by whitespace, 
    and returns a list of these elements.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        list: A list containing the first element from each line of the file.
    """
    list1 = []

    with open(file_path, 'r') as file:
        for line in file:
            element = line.strip().split()
            list1.append(element[0])
    return list1

def check_paths(list_of_lists):
    """
    Validates the existence of file paths provided in a nested list structure.

    This function takes a list of lists containing file paths, flattens it into a single list,
    and checks if each file path corresponds to an existing file on the filesystem. For each
    invalid file path, it prints an error message with the invalid path and keeps a count of
    such occurrences. Finally, it prints the total count of invalid file paths.

    Args:
        list_of_lists (list of list of str): A nested list where each inner list contains file paths as strings.

    Returns:
        None
    """

    paths = flatten_list_comprehension(list_of_lists)
    trues = [os.path.isfile(file) for file in paths]
    counter = 0
    for true, path in zip(trues, paths):
        if true != True:
            print('Non valid route!', path)
            counter +=1
    print('Total number of invalid paths:', counter)

def shuffle_sampler(samplers, epoch):
    """
    Shuffles all the distributed samplers in the provided list for a given epoch.

    This function is typically used in distributed training setups to ensure that 
    the data loading order is randomized for each epoch, improving training performance.

    Args:
        samplers (list): A list of distributed samplers to be shuffled. If the list is empty or None, the function does nothing.
        epoch (int): The current epoch number used to seed the shuffling process.

    Returns:
        None
    """
    if not samplers: # if they are none
        return
    for sampler in samplers:
        sampler.set_epoch(epoch)