# LangChain Using Ollama-Models

## Prerequisites
Ensure you have Python version 3.8 or higher installed on your system.

### Creating and Activating Virtual Environment
1. Install `virtualenv` if not already installed:
    ```bash
    pip install virtualenv
    ```

2. Create a virtual environment named `myenv`:
    ```bash
    virtualenv myenv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```bash
        myenv\Scripts\activate
        ```
    - On Unix or MacOS:
        ```bash
        source myenv/bin/activate
        ```

### Installation Commands for Libraries
Use pip to install the required libraries:
```bash
pip install langchain
pip install openai
pip install huggingface_hub
pip install python_dotenv
pip install streamlit


Ollama Installation

1.Run the following command to install Ollama (Linux):

    curl https://ollama.ai/install.sh | sh

2.Run Ollama

    ollama run llama2

Running the File :

    After completing the above steps, run your Python file using the following command:

        python3 your_file.py


Additional Notes :

    Make sure to replace your_file.py with the actual name of your Python file.
    Ensure that the virtual environment is activated before running the Python file.


