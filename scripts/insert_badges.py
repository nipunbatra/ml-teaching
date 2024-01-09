import os
from nbformat import read, NO_CONVERT, write

def has_github_and_colab_badges(notebook):
    # Check if GitHub and Colab badges are already present in the last cell
    last_cell_index = len(notebook['cells']) - 1
    if last_cell_index >= 0 and 'source' in notebook['cells'][last_cell_index] and ('[![Open In Colab]' in notebook['cells'][last_cell_index]['source'] or '[GitHub]' in notebook['cells'][last_cell_index]['source']):
        return True
    return False

def insert_github_and_colab_badges(notebook_path, gh_path, colab_path, dry_run=False):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as nb_file:
        notebook = read(nb_file, as_version=NO_CONVERT)

    # Check if GitHub and Colab badges are already present
    if not has_github_and_colab_badges(notebook):
        # Create the GitHub and Colab badges markdown
        github_badge_md = f"[![GitHub](https://img.shields.io/badge/GitHub-Open%20In%20GitHub-blue?logo=github)]({gh_path})"
        colab_badge_md = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_path})"

        # Create the markdown cell with badges
        badges_md = f"{github_badge_md}\n\n{colab_badge_md}"
        cell = {'cell_type': 'markdown', 'metadata': {}, 'source': badges_md}

        # Print the action to be taken
        print(f"Action to be taken in {notebook_path}:\nInserting GitHub and Colab badges as the last cell")

        # Print the modified markdown cell
        print(f"\nModified Cell:\n{cell['source']}")

        # Print a separator
        print("\n" + "-"*50 + "\n")

        # If it's not a dry run, modify the notebook
        if not dry_run:
            # Insert the cell as the last cell
            notebook['cells'].append(cell)

            # Write the modified notebook back
            with open(notebook_path, 'w', encoding='utf-8') as nb_file:
                write(notebook, nb_file, as_version=NO_CONVERT)

def dry_run_add_github_and_colab_badges(folder_path, github_repo_link):
    # Iterate through all notebooks in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)

                # Calculate GitHub path for Colab link
                github_path = f"{github_repo_link}/blob/main/{os.path.relpath(notebook_path, folder_path)}"
                colab_path = f"https://colab.research.google.com/github/{username}/{repo_name}/blob/main/{os.path.relpath(notebook_path, folder_path)}"
                # Perform a dry run
                insert_github_and_colab_badges(notebook_path, github_path, colab_path, dry_run=dry_run)

username = "nipunbatra"
repo_name = "ml-teaching"
# Provide your GitHub repo link
github_repo_link = f"https://github.com/{username}/{repo_name}"

# Provide the path to your notebooks folder
notebooks_folder_path = "../notebooks"

dry_run = False

# Call the function for a dry run to see what actions would be taken
dry_run_add_github_and_colab_badges(notebooks_folder_path, github_repo_link)
