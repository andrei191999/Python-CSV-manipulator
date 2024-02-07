### Data Processing Script Documentation

#### Introduction
This Python script is designed to automate and simplify data cleaning, transformation, and analysis tasks for CSV files. It's structured to support both test and live processing environments, allowing users to work on datasets without risking the integrity of the original data. The script is accessible via a command-line interface, providing a user-friendly menu for selecting operations.

#### Setting Up the Project
To set up this Python project from scratch, follow these steps:

1. **Install Python**: 
Ensure that Python 3.x is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **Create a Virtual Environment** (optional but recommended):
   - Navigate to your project directory.
   - Run `python -m venv venv` to create a virtual environment named `venv`.
   - Activate the virtual environment:
     - On Windows, run `.\venv\Scripts\activate`.
     - On macOS/Linux, run `source venv/bin/activate`.

3. **Install Required Packages**: 
Install all required packages by running `pip install -r requirements.txt` in your command line.

5. **Prepare Input/Output Folders**: To organize your Python project for data processing with the `master_script.py` and its associated input/output folders, you can structure the project directory as follows:

    ```
    project_directory/
    │
    ├── master_script.py          # The main script file.
    │
    ├── input_csvs/               # Folder for live input CSV files.
    │   ├── file1.csv
    │   ├── file2.csv
    │   └── ...
    │
    ├── output_csvs/              # Folder for live processed output CSV files.
    │   ├── processed_file1.csv
    │   ├── processed_file2.csv
    │   └── ...
    │
    ├── input_csvs_test/          # Folder for test input CSV files.
    │   ├── test_file1.csv
    │   ├── test_file2.csv
    │   └── ...
    │
    └── output_csvs_test/         # Folder for test processed output CSV files.
        ├── processed_test_file1.csv
        ├── processed_test_file2.csv
        └── ...
    ```

    ### How to Use

    - **Placement of `master_script.py`**: Ensure that `master_script.py` resides in the root of your project directory.
    - **Preparation of CSV Files**: Place your live CSV files in the `input_csvs` folder and your test CSV files in the `input_csvs_test` folder.
    - **Running the Script**: Navigate to your project directory in your command line or terminal and run the script using the Python command:
    ```
    python master_script.py
    ```
    - **Processing Files**: Follow the interactive menu in the command line to select the desired operation. The script will process files from the input folders and save the results in the corresponding output folders.
    - **Reviewing Results**: After processing, check the `output_csvs` or `output_csvs_test` folders for the processed files.

    This structure not only keeps your data organized but also clearly separates test files from live data, reducing the risk of accidental data manipulation.


### Detailed Analysis of Menu Functions
This documentation provides a closer look at each menu function within the Python script designed for data processing. It aims to guide users through various operations, offering insights into their utility and application through examples.

#### 1. Check Individual and Combo Duplicates
   - **Purpose**: Identifies duplicate records based on individual columns or combinations thereof.
   - **Example Use**: If you have a dataset with columns `email` and `phone_number`, this function can find duplicates in each column individually or in records where both `email` and `phone_number` at the same.
   - **How to Use**: Select this option and specify the columns for checking duplicates when prompted.

#### 2. Count Empty Values
   - **Purpose**: Counts the number of empty (null) values in each column across CSV files.
   - **Example Use**: To assess data quality and identify columns with a high number of missing values.
   - **How to Use**: Choose this option from the menu. The script will automatically process all files in the input folder and summarize the counts of empty values per column.

#### 3. Add or Update Column
   - **Purpose**: Adds a new column to the dataset or updates an existing one based on specified operations (e.g., fixed value, copying from another column).
   - **Example Use**: Add a new column named `status` and set its value to "active" for all records.
   - **How to Use**: Select this function and follow the prompts to define the operation, such as `status="active"`.

#### 4. Strip or Remove Spaces from Columns
   - **Purpose**: Cleans up columns by stripping trailing/leading spaces or removing all spaces.
   - **Example Use**: Clean up a `name` column by removing extra spaces around the names.
   - **How to Use**: After selecting this option, specify the column and the type of space removal needed.

#### 5. Split and Order CSVs
   - **Purpose**: Splits large CSV files into smaller chunks and/or orders records based on specified criteria.
   - **Example Use**: Split a large dataset into smaller files with 1000 records each, or sort records by a `date` column.
  - **How to Use**: Indicate the splitting criteria or the column by which to order the data when prompted.

#### 6. Process Mandatory Fields
   - **Purpose**: Identifies rows missing values in specified mandatory fields and segregates them.
   - **Example Use**: Ensure that the `email` and `phone_number` columns do not contain empty values.
   - **How to Use**: Input the mandatory columns, and the script will process the files to identify and isolate rows with missing values.

#### 7. Replace Substring in Column
   - **Purpose**: Finds and replaces specified substrings within a column.
   - **Example Use**: Replace "http://" with "https://" in a `website` column.
   - **How to Use**: Choose this option and specify the column, the substring to find, and the replacement substring.

#### 8. Count Value Occurrences
   - **Purpose**: Counts occurrences of specified values within columns.
   - **Example Use**: Count how many times "pending" appears in a `status` column.
   - **How to Use**: After selecting, enter the column name and the value to count.

#### 9. Format Numbers or Dates in Column
   - **Purpose**: Applies formatting to numbers or dates within a specified column.
   - **Example Use**: Convert a `date` column to a "YYYY-MM-DD" format.
   - **How to Use**: Specify the column and the desired format.

#### 10. Rename Column Header
   - **Purpose**: Renames one or more column headers.
   - **Example Use**: Rename `phonenumber` to `phone_number`.
   - **How to Use**: Input the old column name(s) and the new name(s) when prompted.

#### Helper Functions
   - **`format_number`**: Formats numbers by removing spaces and standardizing decimal separators. Returns the formatted number or the original value if formatting fails.
   - **`save_issue_summaries` & `save_multiple_issues_summary`**: Save detailed reports of issues found during data processing, such as missing values or data inconsistencies, into separate CSV files for easy review.
   - **`parse_operation`**: Parses a string representing an operation (e.g., column transformations, substring extractions) and applies it to a DataFrame. Supports fixed value assignment, date manipulation, substring operations, and concatenations.
   - **`parse_user_input_for_columns_and_groups` & `get_column_names`**: Facilitate user interaction by allowing selection of columns for processing based on user input. Supports individual columns, groups of columns, and the addition of new columns.
   - **`sanitize_filename`**: Ensures filenames are valid and safe for the file system by removing or replacing problematic characters.
   - **`read_csv_with_dtypes`**: Reads a CSV file into a pandas DataFrame while applying specified data types or converters to ensure correct data handling.
   - **`get_data_types`**: Guides the user in specifying the data types for each column in a DataFrame, improving the accuracy of subsequent data processing tasks.
   - **`preprocess_float`**: A converter function used to preprocess float values by removing spaces before conversion, aiding in data type consistency

#### Usage Examples
After setting up the input/output folders, navigate to the script's directory in your command line and run the script. Use the menu to select an operation and follow the prompts to specify input/output folders, data types for columns (if necessary), and other operation-specific options. For detailed usage examples for each function, refer to the inline comments and prompts within the script.

#### Conclusion
This script is designed to be a comprehensive tool for data processing, offering flexibility for both development/testing and production use. By automating routine data cleaning and transformation tasks, it aims to save time and reduce the potential for manual errors, making it a valuable addition to any data analyst's toolkit.

Remember to always back up your data before performing live processing to prevent accidental data loss or corruption.
