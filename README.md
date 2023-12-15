# Coursework C | George Madeley

|University of Bath|
| :- |
|Coursework C|
|EE40098: Computational Intelligence|

|<>George Madeley 12-15-2023|
| :- |

## Installation

To install the necessary packages and dependencies for running the program, we will be using pipenv. Pipenv is a tool that combines pip package management and virtual environments.

1. Make sure you have pipenv installed on your system. If not, you can install it by running the following command:

    ```CMD
    pip install pipenv
    ```

2. Once pipenv is installed, navigate to the project directory in your terminal.

3. Run the following command to install the packages specified in the Pipfile:

    ```CMD
    pipenv install
    ```

    This will create a virtual environment and install all the required packages and their specific versions mentioned in the Pipfile.

4. To activate the virtual environment, run the following command:

    ```CMD
    pipenv shell
    ```

    This will activate the virtual environment and allow you to run the program with the installed packages.

## Running the Program

You can run the program by either calling `main.py` or using the launch configuration in `.vscode/launch.json`. The code will run through each of the .mat files found in ./data.

NOTE: All D1 to D6.mat files must be present for the code to run.

### Calling main.py

1. Open a terminal or command prompt.

2. Navigate to the project directory.

3. Run the following command:

    ```CMD
    pipenv run python ./scripts/main.py
    ```

    This will execute the `main.py` script using pipenv and run the program.

### Using the Launch Configuration

1. Open Visual Studio Code.

2. Open the Run and Debug panel.

3. In the top dropdown menu, select the desired launch configuration.

4. Click the green play button to start debugging and run the program.
