import subprocess

def list_of_models():
    try:
        # Run the command and capture its output
        output = subprocess.check_output(['ollama', 'list'], universal_newlines=True)
        # Split the output by lines
        lines = output.strip().split('\n')
        # Print header
        print("{:<30}".format('NAME'))
        # Iterate over each line and print the details
        for line in lines[1:]:
            columns = line.split()
            name = columns[0]
            image_id = columns[1]
            size = columns[2]
            modified = ' '.join(columns[3:])
            print("{:<30}".format(name))
    except subprocess.CalledProcessError as e:
        print("Error running ollama command:", e)

# Call the function to list images
list_of_models()
