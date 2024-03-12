import os
import re
import unicodedata
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from multiprocessing import Manager
from tabulate import tabulate
from functools import partial

# Global variables
global input_folder, output_folder, in_date_format, boolean_format, out_date_format, sender_name, timestamp, numeric_format, incorrect_rows_accumulator, confirmed_data_types
in_date_format = out_date_format = numeric_format = boolean_format = ""
input_folder = output_folder = sender_name = ""
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
incorrect_rows_accumulator = []
confirmed_data_types = {}


def main():
    global sender_name, timestamp, input_folder, output_folder
    print("Welcome to the CSV Utility Tool")
    print("Make sure to place this script in a new folder.")
    print("Create two folders in this directory: 'input_csvs_test' and 'input_csvs'.")
    print(
        "All output CSVs will be saved in 'output_csvs_test' or 'output_csvs' based on your choice."
    )

    # Ask user if they want to run in test mode or live mode
    mode = input("Select mode (test/live): ").strip().lower()
    # Get sender name from user
    sender_name = input("\nEnter the sender name to prefix output files: ").strip()

    # Check for mode and set input and output folders
    if any(char in mode for char in ["t", "s"]):
        input_folder = "input_csvs_test"
        output_folder = "output_csvs_test"
    elif any(char in mode for char in ["l", "v"]):
        input_folder = "input_csvs"
        output_folder = "output_csvs"
    else:
        print("Invalid mode selected. Exiting.")
        return

    # Inform the user about data type options and get user-defined types
    get_data_types_and_clean()
    # print(confirmed_data_types)

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        print("\nMenu:")
        print("1. Check and extract duplicates from columns")
        print("2. Count empty values in columns")
        print("3. Add/update a column")
        print("4. Strip/remove spaces from columns")
        print("5. Order by column and rename csvs")
        print("6. Process mandatory fields")
        print("7. Replace substring in a column")
        print("8. Count occurences of given value OR all unique values in column")
        print("9. Format dates OR numbers (dot (.) as decimal sep, no thousand sep) ")
        print("10. Rename a column header")
        print("0. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            check_individual_and_combo_duplicates()
        elif choice == "2":
            count_empty_values()
        elif choice == "3":
            add_or_update_column()
        elif choice == "4":
            strip_or_remove_spaces_from_columns()
        elif choice == "5":
            split_and_order_csvs()
        elif choice == "6":
            process_mandatory_fields()
        elif choice == "7":
            replace_substring_in_column()
        elif choice == "8":
            count_value_occurrences()
        elif choice == "9":
            format_numbers_or_dates_in_column()
        elif choice == "10":
            rename_column_header()
        elif choice == "0":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")


def rename_column_header():
    column_prompt = "Enter the column number/name to rename: "
    current_column_header, _ = get_column_names(
        None, column_prompt, return_single_col=True
    )

    if current_column_header is None:
        print("Invalid column selection.")
        return

    print(f"Current header for column: {current_column_header}")
    new_column_header = input("Enter the new header name: ").strip()

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            sanitized_filename = sanitize_filename(filename)
            output_file_path = os.path.join(output_folder, sanitized_filename)

            try:
                df = pd.read_csv(file_path)
                df = df.rename(columns={current_column_header: new_column_header})
                df.to_csv(output_file_path, index=False)
                print(f"Renamed column header in file: {sanitized_filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(
        f"Column header renaming completed. Updated files are saved in {output_folder}"
    )


# This script section applies the format_number function to a specific column in all CSV files in a folder
def format_numbers_or_dates_in_column():
    column_prompt = "Enter the column number/name to format numbers or dates: "
    column_name, existing_columns = get_column_names(
        None,
        column_prompt,
        allow_new_columns=False,
        return_single_col=True,
    )

    if column_name is None:
        return  # Exit if no valid column name is provided

    total_formatted_count = 0  # Counter for the total number of formatted values
    sample_before_after = []  # To store before and after formatting samples

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            try:
                df = pd.read_csv(
                    file_path, dtype=confirmed_data_types, low_memory=False
                )
                file_formatted_count = 0  # Counter for the current file

                if column_name in df.columns:
                    # Apply formatting to each value in the column
                    results = []
                    changes = []
                    for value in df[column_name]:
                        formatted_value, changed = format_number(value)
                        results.append(formatted_value)
                        changes.append(changed)
                        if changed:
                            file_formatted_count += 1
                            total_formatted_count += 1
                            if len(sample_before_after) < 3:
                                sample_before_after.append((value, formatted_value))

                    # Update the column with formatted results
                    df[column_name] = results

                    # Save the updated DataFrame to a new CSV
                    df.to_csv(output_file_path, index=False)
                    print(
                        f"Processed file '{filename}' with {file_formatted_count} values formatted."
                    )

                else:
                    print(f"Column '{column_name}' not found in file: {filename}")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Output the total count and samples
    print(
        f"\nTotal number of values formatted across all files: {total_formatted_count}"
    )
    print("Sample values before and after formatting:")
    for original, formatted in sample_before_after:
        print(f"Original: {original}, Formatted: {formatted}")


# This function will be used within the script to format numbers in a column
def format_number(value):
    # Check if the value is NaN or None, return as is
    if pd.isna(value) or value is None:
        return value, False  # No change made

    # Convert to string for processing
    str_value = str(value)

    # Store original for comparison
    original_value = str_value

    # Detect negative numbers and remove minus sign for processing
    is_negative = str_value.startswith("-")
    if is_negative:
        str_value = str_value[1:]

    # Remove all spaces and replace common decimal separators with a period
    str_value = str_value.replace(" ", "").replace(",", ".")

    # Detect multiple periods (as thousand separators)
    if str_value.count(".") > 1:
        parts = str_value.split(".")
        # Join all parts without the last (decimal part) to remove thousand separators
        str_value = "".join(parts[:-1]) + "." + parts[-1]

    # Reapply the negative sign if the number was negative
    formatted_value = "-" + str_value if is_negative else str_value

    # Return the value as a float if it's a valid number, else original
    try:
        float_value = float(formatted_value)
        if original_value != formatted_value:
            return float_value, True  # Change made
        else:
            return float_value, False  # No change made
    except ValueError:
        return value, False  # No change made


def split_and_order_csvs():
    column_prompt = "Enter the column number/name to order by: "
    order_column, existing_columns = get_column_names(
        None,
        column_prompt,
        allow_new_columns=False,
        return_single_col=True,
    )

    if order_column is None:
        return  # Exit if no valid column name is provided

    all_data = pd.DataFrame()
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_csv(
                    file_path, dtype=confirmed_data_types, low_memory=False
                )
                all_data = pd.concat([all_data, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    all_data = all_data.sort_values(by=order_column)

    # Split into chunks of 20,000 rows
    chunk_size = 20000
    num_chunks = len(all_data) // chunk_size + (1 if len(all_data) % chunk_size else 0)

    for i in range(num_chunks):
        chunk = all_data.iloc[i * chunk_size : (i + 1) * chunk_size]
        min_val = chunk[order_column].iloc[0]
        max_val = chunk[order_column].iloc[-1]

        # Include chunk index in filename to avoid overwriting
        sanitized_filename = sanitize_filename(
            f"{sender_name}_{order_column}_{min_val}_to_{max_val}_chunk{i+1}_{timestamp}.csv"
        )
        chunk_file_path = os.path.join(output_folder, f"{sanitized_filename}")
        chunk.to_csv(chunk_file_path, index=False)
        print(f"Saved chunk {i+1} to {chunk_file_path}")

    print("Splitting and ordering process completed.")


def count_value_occurrences():
    # Allow selection of multiple columns
    column_prompt = "Enter column names/numbers to count value/substring occurrences (separated by commas): "
    column_names, _ = get_column_names(
        None,
        column_prompt,
        allow_new_columns=False,
        return_single_col=False,
    )

    if not column_names:
        return  # Exit if no valid column names are provided

    input_value = input(
        'Enter a specific value or substring to search (enclose substrings in ""): '
    ).strip()

    # Determine if the input is a substring or a direct value
    is_substring = input_value.startswith('"') and input_value.endswith('"')
    value_to_search = input_value.strip('"') if is_substring else input_value

    # Counters for each column
    occurrence_counters = {col: 0 for col in column_names}
    rows_with_searched_values = {col: pd.DataFrame() for col in column_names}

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_csv(
                    file_path, dtype=confirmed_data_types, low_memory=False
                )
                for col in column_names:
                    if col in df.columns:
                        mask = (
                            df[col].apply(lambda x: value_to_search in str(x))
                            if is_substring
                            else df[col] == value_to_search
                        )
                        rows_with_searched_value = df[mask]
                        occurrence_counters[col] += rows_with_searched_value.shape[0]
                        rows_with_searched_values[col] = pd.concat(
                            [rows_with_searched_values[col], rows_with_searched_value],
                            ignore_index=True,
                        )
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    for col, rows in rows_with_searched_values.items():
        if not rows.empty:
            sanitized_filename = sanitize_filename(
                sender_name
                + "_rows_with_"
                + col
                + "_"
                + value_to_search
                + "_"
                + timestamp
                + ".csv"
            )
            output_file_path = os.path.join(output_folder, sanitized_filename)
            rows.to_csv(output_file_path, index=False)
            print(
                f"Rows with '{value_to_search}' in column '{col}' saved to {output_file_path}"
            )

    # Print occurrence counts
    for col, count in occurrence_counters.items():
        print(f"Occurrences in '{col}': {count}")

    if all(count == 0 for count in occurrence_counters.values()):
        print(f"No rows found with '{value_to_search}' in the selected columns.")


def replace_substring_in_column():
    column_prompt = "Enter the column number/name in which to replace the substring: "
    column_name, existing_columns = get_column_names(
        None,
        column_prompt,
        allow_new_columns=False,
        return_single_col=True,
    )

    if column_name is None:
        return  # Exit if no valid column name is provided

    search_substring = input("Enter the substring to search for: ")
    replace_substring = input("Enter the substring to replace with: ")

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            try:
                df = pd.read_csv(
                    file_path, dtype=confirmed_data_types, low_memory=False
                )

                if column_name in df.columns:
                    df[column_name] = (
                        df[column_name]
                        .astype(str)
                        .str.replace(search_substring, replace_substring)
                    )
                    df.to_csv(output_file_path, index=False)
                    print(f"Processed file: {filename}")
                else:
                    print(f"Column '{column_name}' not found in file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print("Substring replacement completed.")


def process_mandatory_fields():

    def save_issue_summaries(
        issue_dfs, issue_counts, sender_name, timestamp, output_folder
    ):
        for col, df_col_issues in issue_dfs.items():
            if not df_col_issues.empty:
                issues_file_path = sanitize_filename(
                    output_folder
                    + sender_name
                    + "_issues_"
                    + col
                    + "_"
                    + timestamp
                    + ".csv"
                )
                df_col_issues.to_csv(issues_file_path, index=False)
                print(
                    f"Issues in column '{col}' saved to {issues_file_path}. Total issues: {issue_counts[col]}"
                )

    def save_multiple_issues_summary(
        multiple_issue_records, sender_name, timestamp, output_folder
    ):
        if multiple_issue_records:
            multiple_issues_df = pd.concat(multiple_issue_records, ignore_index=True)
            multi_issues_file_path = sanitize_filename(
                output_folder + sender_name + "_multiple_issues_" + timestamp + ".csv"
            )
            multiple_issues_df.to_csv(multi_issues_file_path, index=False)
            print(
                f"Rows with multiple issues saved to {multi_issues_file_path}. Total rows: {len(multiple_issues_df)}"
            )

    global sender_name
    processed_files_count = 0
    failed_files = []
    multiple_issue_records = []

    files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    total_files = len(files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    column_prompt = "Enter mandatory column names/numbers (separated by commas): "
    mandatory_columns, existing_columns = get_column_names(
        None,
        column_prompt,
        allow_new_columns=False,
        return_single_col=False,
    )

    if not mandatory_columns:
        print("No mandatory columns specified. Exiting.")
        return

    issue_dfs = {col: pd.DataFrame() for col in mandatory_columns}
    issue_counts = {col: 0 for col in mandatory_columns}

    print(f"Processing {total_files} files...")
    with tqdm(total=total_files, desc="Initializing...") as pbar:
        for filename in files:
            # Update the progress bar's description to include the current file being processed
            pbar.set_description(f"Processing {filename}")
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_csv(
                    file_path, dtype=confirmed_data_types, low_memory=False
                )

                file_issue_counts = {
                    col: 0 for col in mandatory_columns
                }  # Reset per-file issue counters

                has_issues = df[mandatory_columns].isnull().any(axis=1)
                multiple_issues = df[mandatory_columns].isnull().sum(axis=1) > 1
                no_issues = ~has_issues

                # Save rows without any issues
                if no_issues.any():
                    df_no_issues = df[no_issues]
                    df_no_issues.to_csv(
                        os.path.join(output_folder, filename), index=False
                    )

                if has_issues.any():
                    for col in mandatory_columns:
                        col_issues = df[col].isnull()
                        if col_issues.any():
                            issue_dfs[col] = pd.concat(
                                [issue_dfs[col], df[col_issues]], ignore_index=True
                            )
                            count = col_issues.sum()
                            issue_counts[col] += count
                            file_issue_counts[col] += count

                    # Log detailed file issues
                    issue_details = "; ".join(
                        [
                            f"{col}: {count} issue(s)"
                            for col, count in file_issue_counts.items()
                            if count > 0
                        ]
                    )
                    print(
                        f"Processed {filename} - Detected issues in columns - {issue_details}"
                    )

                if multiple_issues.any():
                    multiple_issue_df = df[multiple_issues]
                    multiple_issue_records.append(multiple_issue_df)
                    print(
                        f"Processed {filename} - Detected {multiple_issues.sum()} rows with multiple issues"
                    )
                processed_files_count += 1

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                failed_files.append(filename)
            finally:
                # Update the progress bar after each file, regardless of success or failure
                pbar.update(1)

    # Save issue summaries
    save_issue_summaries(issue_dfs, issue_counts, sender_name, timestamp, output_folder)
    save_multiple_issues_summary(
        multiple_issue_records, sender_name, timestamp, output_folder
    )

    # Final summary
    print("\nMandatory fields processing completed.")
    print(f"Processed {total_files} files.")
    # After processing all files, output summary information
    if failed_files:
        print(f"Warning: Encountered errors with {len(failed_files)} file(s).")
        for failed_file in failed_files:
            print(f"Failed to process file: {failed_file}")

    print(f"Successfully processed {processed_files_count} files out of {total_files}.")
    for col, count in issue_counts.items():
        print(f"Total issues detected in column '{col}': {count}.")


def strip_or_remove_spaces_from_columns():
    column_prompt = "Enter column names/numbers to modify (separated by commas), or press enter to process all columns: "

    column_names, _ = get_column_names(
        None,
        column_prompt,
        allow_new_columns=False,
        return_single_col=False,
    )

    if column_names is None:
        return  # Exit the function if no valid column names are provided

    operation_prompt = "Choose operation - 'strip' to remove leading/trailing spaces, 'remove' to remove all spaces: "
    operation = input(operation_prompt).strip().lower()

    if operation not in ["strip", "remove"]:
        print("Invalid operation selected. Exiting.")
        return

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            try:
                df = pd.read_csv(
                    file_path, dtype=confirmed_data_types, low_memory=False
                )

                for col in column_names:
                    if col in df.columns and df[col].dtype == "object":
                        if operation == "strip":
                            df[col] = df[col].str.strip()
                        elif operation == "remove":
                            df[col] = df[col].str.replace(" ", "", regex=True)

                df.to_csv(output_file_path, index=False)
                print(f"Processed file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(
        f"Operation completed. Modified spaces in columns in files saved to {output_folder}"
    )


def parse_operation(operation, df):
    # Helper function for substring extraction
    def get_substring(value, start, end):
        if start == "-":
            start = 0
        else:
            start = int(start)
            if start < 0:
                # When start is negative, the end should be None to get the last 'start' characters
                end = None
                start = (
                    len(value) + start
                )  # Calculate the correct start index for a negative value
            else:
                # Adjust for zero-based indexing for positive indices
                start -= 1

        if end != "-" and end is not None:
            end = int(end)
            if start >= 0:
                end += start  # Adjust the end index based on the start index

        return value[start:end]

    fixed_string_match = re.match(r"^\"([^\"]+)\"$", operation)
    # Date operation (e.g., 'col+10y')
    date_match = re.match(r"(\w+)\+(\d+)([ymd])", operation)
    # Substring operation pattern (e.g., 'col(1,4)')
    substring_match = re.match(r"(\w+)\((-?\d{1,2}|-),?(\d{1,2}|-)?\)", operation)

    # Add a debugging statement to print the operation
    # print(f"Debug: Parsing operation: {operation}")

    if date_match:
        col, offset, unit = date_match.groups()
        offset = int(offset)
        if col not in df.columns:
            return None, f"Column '{col}' not found for date operation."
        if unit == "y":
            return (
                df[col].apply(
                    lambda x: pd.to_datetime(x) + relativedelta(years=offset)
                ),
                None,
            )
        elif unit == "m":
            return (
                df[col].apply(
                    lambda x: pd.to_datetime(x) + relativedelta(months=offset)
                ),
                None,
            )
        elif unit == "d":
            return (
                df[col].apply(lambda x: pd.to_datetime(x) + relativedelta(days=offset)),
                None,
            )

    # Concatenation with substring operation (e.g., 'col1(1,4)+col2+"_text"')
    elif "+" in operation:
        parts = re.findall(
            r"(\w+\(-?\d+,-?\d+\)|\w+\(-?\d+,-\)|\w+\(-,-?\d+\)|\w+|\"[^\"]+\")",
            operation,
        )
        combined = ""
        for part in parts:
            substring_match_concat = re.match(
                r"(\w+)\((-?\d{1,2}|-),?(\d{1,2}|-)?\)", part
            )
            if substring_match_concat:
                col, start, end = substring_match_concat.groups()
                if col in df.columns:
                    combined += df[col].apply(
                        lambda x: get_substring(str(x), start, end)
                    )
                else:
                    return None, f"Column '{col}' not found for substring operation."
            elif part.startswith('"') and part.endswith('"'):
                combined += part.strip('"')
            elif part in df.columns:
                combined += df[part].astype(str)
            else:
                return None, f"Column '{part}' not found for concatenation."
        return combined, None

    elif substring_match:
        col, start, end = substring_match.groups()
        # Debugging output
        # print(f"Debug: Substring operation found. Column: {col}, Start: {start}, End: {end}")

        if col in df.columns:
            return df[col].apply(lambda x: get_substring(str(x), start, end)), None
        else:
            return None, f"Column '{col}' not found for substring operation."

    # Handle column-to-column copy (e.g., 'to_archive=to_presentment')
    elif operation in df.columns:
        return df[operation], None

    # Handle fixed string assignment (e.g., 'to_archive="true"')
    elif fixed_string_match:
        value = fixed_string_match.group(1)
        return value, None

    # Debugging output for when no valid format is found
    # print(f"Debug: No valid operation format found for: {operation}")
    return None, "Invalid operation format."


# Type Conversion: Automatically convert non-string data to strings before concatenation, or handle different data types explicitly.
# Robust Error Handling: Add more checks and error messages to handle cases like invalid indices, non-existent columns, or unsupported operations.
def add_or_update_column():
    column_prompt = "Enter the column number to update, '-1' to add a new column, or column name directly: "
    column_name, _ = get_column_names(
        None,
        column_prompt,
        allow_new_columns=True,
        return_single_col=True,
    )

    print("Enter the operation. Here are some examples:")
    print("-" * 50)  # Print a divider for better visual separation

    print(f'  1. Set a fixed value: {column_name}="True"')
    print(f"     (Sets all rows in '{column_name}' to 'True')")

    print(f"  2. Copy from another column: {column_name}=doc_date")
    print(f"     (Copies values from 'doc_date' to '{column_name}')")

    print(f"  3. Extract substring: {column_name}=doc_type(2,5)")
    print(
        f"     (Takes characters 2 to 5 from 'doc_type' and puts them in '{column_name}')"
    )

    print(f"  4. Extract last N characters: {column_name}=doc_type(-5)")
    print(
        f"     (Takes the last 5 characters from 'doc_type' and puts them in '{column_name}')"
    )

    print(f"  5. Add years to a date: {column_name}=intake_date+10y")
    print(
        f"     (Adds 10 years to dates in 'intake_date' and stores them in '{column_name}')"
    )

    print(f'  6. Concatenate columns: {column_name}=doc_uuid+"_"+doc_id')
    print(
        f"     (Concatenates 'doc_uuid' and 'doc_id' with an underscore and puts it in '{column_name}')"
    )

    print(
        f'  7. Concatenate with substring: {column_name}=doc_uuid(1,4)+"_"+to_archive(5,-)'
    )
    print(
        f"     (Concatenates substrings of 'doc_uuid' and 'to_archive' and puts it in '{column_name}')"
    )

    print(
        f'  8. Concatenate with last N characters: {column_name}=doc_uuid(-5)+"_"+doc_id'
    )
    print(
        f"     (Concatenates the last 5 characters of 'doc_uuid' with 'doc_id' and puts it in '{column_name}')"
    )

    print("-" * 50)  # Another divider for separation
    print("Note:")
    print("  - The concatenation operations do NOT work with column numbers.")
    print('  - Strings need to be enclosed in double quotes ("").')
    print("  - Unquoted text is treated as column names.")

    operation = input(f"Enter the operation for '{column_name}'=").strip()

    files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    total_files = len(files)

    print(f"Processing {total_files} files...")
    with tqdm(total=total_files, desc="Initializing...") as pbar:
        for filename in files:
            # Update the progress bar's description to include the current file being processed
            pbar.set_description(f"Processing {filename}")
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            try:
                df = pd.read_csv(
                    input_file_path, dtype=confirmed_data_types, low_memory=False
                )
                operation_result, error_message = parse_operation(operation, df)
                if error_message:
                    print(error_message)
                    pbar.update(
                        1
                    )  # Ensure the progress bar updates even if returning due to error
                    continue

                if operation_result is not None:
                    df[column_name] = operation_result
                    df.to_csv(output_file_path, index=False)
                else:
                    print("Error in operation. Please check the format and try again.")
                    pbar.update(
                        1
                    )  # Ensure the progress bar updates even if returning due to error
                    continue
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

            pbar.update(1)  # Update the progress bar after processing each file

    print(f"Operation completed. Updated files are saved in {output_folder}")


def parse_user_input_for_columns_and_groups():
    column_prompt = "Enter columns and groups to check for duplicates (e.g., '1,5,doc_date,[1,3,doc_date]'): "
    column_names, _ = get_column_names(
        None,
        column_prompt,
        allow_new_columns=False,
        return_single_col=False,
    )

    parsed_columns = []
    for col in column_names:
        if isinstance(col, str) and col.startswith("[") and col.endswith("]"):
            # It's a group of columns; parse it into a tuple
            group = col.strip("[]").split(",")
            group_column_names, _ = get_column_names(
                ",".join(group),
                "",
                allow_new_columns=False,
                return_single_col=False,
            )
            if group_column_names:
                parsed_columns.append(tuple(group_column_names))
        else:
            # It's an individual column
            parsed_columns.append(col)

    return parsed_columns


def check_individual_and_combo_duplicates():

    # Get column names or groups to check for duplicates
    column_names = parse_user_input_for_columns_and_groups()

    if not column_names:
        print("No columns or groups to check. Exiting.")
        return

    combined_df = pd.DataFrame()
    file_frames = {}
    files_with_duplicates = set()  # Track files that have duplicates

    # Combine all files into one DataFrame and store individual DataFrames
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)

            try:
                df = pd.read_csv(
                    file_path, dtype=confirmed_data_types, low_memory=False
                )
                df["_original_filename"] = filename
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                file_frames[filename] = df
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Check for duplicates
    for col in column_names:
        col_key = str(col) if isinstance(col, tuple) else col
        duplicates = combined_df[
            combined_df.duplicated(
                subset=col if isinstance(col, str) else list(col), keep=False
            )
        ]
        if not duplicates.empty:
            for filename in file_frames:
                file_duplicates = duplicates[
                    duplicates["_original_filename"] == filename
                ]
                if not file_duplicates.empty:
                    # Identify duplicates in the original file by comparing column values
                    original_duplicates_indices = file_frames[filename][
                        file_frames[filename]
                        .isin(file_duplicates.to_dict("list"))
                        .all(axis=1)
                    ].index
                    file_frames[filename] = file_frames[filename].drop(
                        original_duplicates_indices
                    )
                    files_with_duplicates.add(filename)

    # Save cleaned files and indicate their status
    for filename, df in file_frames.items():
        df.drop("_original_filename", axis=1, inplace=True)
        output_file_path = os.path.join(output_folder, filename)
        df.to_csv(output_file_path, index=False)
        if filename in files_with_duplicates:
            print(f"Cleaned file saved (duplicates removed): {output_file_path}")
        else:
            print(f"File copied as-is (clean): {output_file_path}")

    # Save duplicates to separate CSV files per column/group
    for col in column_names:
        col_key = str(col) if isinstance(col, tuple) else col
        duplicates = combined_df[
            combined_df.duplicated(
                subset=col if isinstance(col, str) else list(col), keep=False
            )
        ]
        duplicates_copy = duplicates.copy()
        duplicates_copy.drop("_original_filename", axis=1, inplace=True)

        if not duplicates_copy.empty:
            output_file_name = f"{sender_name}_duplicates_{col_key}_{timestamp}.csv"
            output_file_path = os.path.join(output_folder, output_file_name)
            duplicates_copy.to_csv(output_file_path, index=False)
            print(
                f"Duplicates for column/group '{col_key}' saved to {output_file_path}"
            )

    print("\nDuplicate checking process completed.")


def count_empty_values():
    placeholders = ["", " ", "nan", "-", "_", ".", "na", "null"]

    column_prompt = "Enter column names/numbers to count empty values (separated by commas), or press enter to check all columns: "
    column_names, existing_columns = get_column_names(
        None,
        column_prompt,
        allow_new_columns=False,
        return_single_col=False,
    )

    if column_names is None:
        return  # Exit the function if no valid column names are provided

    empty_counts = {col: 0 for col in column_names}
    total_fields_analyzed = {col: 0 for col in column_names}
    rows_with_empty_values = {col: pd.DataFrame() for col in column_names}

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)

            try:
                df = pd.read_csv(
                    file_path, dtype=confirmed_data_types, low_memory=False
                )

                for col in column_names:
                    if col in df.columns:
                        col_mask = df[col].map(
                            lambda x: str(x).lower() in placeholders or pd.isnull(x)
                        )
                        empty_counts[col] += col_mask.sum()
                        total_fields_analyzed[col] += len(df[col])
                        if col_mask.any():
                            rows_with_empty_values[col] = pd.concat(
                                [rows_with_empty_values[col], df[col_mask]],
                                ignore_index=True,
                            )
                print(f"Processed file: {filename}")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    # First, display columns with no empty or placeholder values
    for col in column_names:
        if col in existing_columns and empty_counts[col] == 0:
            print(f"No empty or placeholder values found in column '{col}'.")

    # Next, display columns with empty or placeholder values
    for col in column_names:
        if col in rows_with_empty_values and not rows_with_empty_values[col].empty:
            # Prefix the sender name to the filename
            sanitized_filename = sanitize_filename(
                sender_name + "_empty_values_" + col + "_" + timestamp + ".csv"
            )
            output_file_path = os.path.join(output_folder, f"{sanitized_filename}")
            rows_with_empty_values[col].to_csv(output_file_path, index=False)

            print(
                f"\nTotal number of fields analyzed in column '{col}': {total_fields_analyzed[col]}"
            )
            print(
                f"Number of empty or placeholder values in column '{col}': {empty_counts[col]}"
            )
            print(f"\nFirst 3 rows with empty or placeholder values in column '{col}':")
            print(rows_with_empty_values[col].head(3))
            print(
                f"Rows with empty or placeholder values for '{col}' saved to {output_file_path}"
            )
        elif col in existing_columns:
            print(f"No empty or placeholder values found in column '{col}'.")


def get_column_names(
    input_cols,
    user_prompt,
    allow_new_columns=False,
    return_single_col=False,
):
    existing_columns = []

    # Read the first file to get column names
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_csv(file_path, nrows=0)  # Read no data, only headers
                existing_columns = df.columns
                break
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                return None, None

    if len(existing_columns) == 0:
        print("No CSV files found or unable to read columns.")
        return None, existing_columns

    print("Existing columns:")
    for idx, col in enumerate(existing_columns, 1):
        print(f"  {idx}. {col}")

    print("\nSelect a column by typing its name or number.")

    if input_cols == None:
        column_input = input(user_prompt).strip()
    else:
        column_input = input_cols

    if column_input == "":
        return existing_columns, existing_columns

    column_names = []
    valid_input = True
    for col in column_input.split(","):
        col = col.strip()
        if col == "-1" and allow_new_columns:
            new_column_name = input("Enter the name of the new column: ").strip()
            print(f"Adding new column: {new_column_name}")
            return new_column_name, existing_columns
        elif col.isdigit():
            column_number = int(col) - 1
            if 0 <= column_number < len(existing_columns):
                column_names.append(existing_columns[column_number])
            else:
                print(f"Invalid column number: {col}")
                valid_input = False
                break
        elif col in existing_columns:
            column_names.append(col)
        # Check if the column name exists in the data; if not, confirm adding a new column
        elif allow_new_columns and col not in existing_columns:
            confirmation = (
                input(
                    f"The column '{col}' does not exist. Do you want to add it as a new column? (yes [Y]/ no [N]): "
                )
                .strip()
                .lower()
            )
            if confirmation != "y":
                print("Operation cancelled.")
                return None, existing_columns
            print(f"Adding new column: {col}")
            return col, existing_columns
        else:
            print(f"Invalid column name: {col}")
            valid_input = False
            break

    if return_single_col:
        return column_names[0] if column_names else None, existing_columns
    else:
        return column_names if valid_input else None, existing_columns


def sanitize_filename(name):
    """Sanitize a string to be used as a valid filename. Removes or replaces characters that are not
    allowed in filenames, trims whitespace, and avoids names that could be problematic for a file system.
    """
    # Remove leading and trailing whitespace
    name = name.strip()

    # Replace slashes with underscores to prevent directory traversal
    name = name.replace("/", "_").replace("\\", "_")

    # Replace colons with underscores (Windows filenames cannot contain colons)
    name = name.replace(":", "_")

    # Normalize unicode characters to their closest ASCII equivalent
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")

    # Remove any characters that are not alphanumeric, underscores, or dots
    name = re.sub(r"[^\w\.\-]", "_", name)

    # Avoid file names that could cause issues in Windows
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    # Strip reserved names
    if name.upper() in reserved_names:
        name = "_" + name

    # Remove leading or trailing dots (which can hide file extensions or be problematic on Unix systems)
    name = name.strip(".")

    # If after all replacements the name is empty (e.g., because it was all invalid chars), replace with underscore
    if not name:
        name = "_"

    return name


def confirm_or_edit_data_types(
    accumulated_probabilities,
    column_order,
    empty_values_count,
    total_counts_accumulator,
):
    """
    Allow the user to confirm or edit the inferred data types with explanations and a help option.

    Parameters:
    - accumulated_probabilities: Dict with column names as keys and dict of data type probabilities as values.
    - column_order: List of column names in the order they appear in the input files.
    """
    confirmed_data_types = {}
    dtype_map = {
        "d": "date",
        "f": "float",
        "i": "integer",
        "b": "boolean",
        "s": "string",
    }

    print("\nYou can now confirm or edit the inferred data types for each column.")
    print(
        "Enter 'd' for date, 'f' for float, 'i' for integer, 'b' for boolean, 's' for string."
    )
    print(
        "Press Enter to accept the current type, or type 'help' for more information on data types.\n"
    )

    for col in column_order:
        total_analyzed = total_counts_accumulator.get(col, 0)
        empty_count = empty_values_count.get(col, 0)

        # Construct the string representation of data types
        dtype_probs = accumulated_probabilities.get(col, {})
        dtype_str = (
            "STRING (defaulted due to empty column)"
            if total_analyzed == 0
            else ", ".join(
                f"{dtype.upper()} with {prob*100:.2f}% certainty"
                for dtype, prob in dtype_probs.items()
                if prob > 0
            )
        )

        while True:  # Keep asking until a valid response is given
            response = (
                input(f'Column "{col}" inferred as {dtype_str} -> ').strip().lower()
            )

            if response == "help":
                print("\nHelp on choosing the correct data type:")
                print("- 'd' (Date): Recognizes and allows operations on dates.")
                print(
                    "- 'f' (Float): Converts integers to floats, adding decimal points where appropriate."
                )
                print(
                    "- 'i' (Integer): Keeps numbers as integers, suitable for countable quantities."
                )
                print(
                    "- 'b' (Boolean): Recognizes 'True' and 'False' values. Choosing this will case-normalize the values to either 'True' or 'False'."
                )
                print(
                    "- 's' (String): Treats the column as plain text. Choosing this prevents numerical or date operations on the column values.\n"
                )
                continue

            if response in dtype_map:
                confirmed_data_types[col] = dtype_map[response]
                action = "changed to" if total_analyzed > 0 else "defaulted to"
                print(f"{col}: Data type {action} {dtype_map[response].upper()}.\n")
                break

            if response == "":
                # Accept the most probable inferred data type if no input is provided or default to string for completely empty columns
                if all(prob == 0 for prob in dtype_probs.values()):
                    most_probable_dtype = "string"
                elif dtype_probs:
                    # Determine the most probable data type based on the highest probability
                    most_probable_dtype = max(dtype_probs.items(), key=lambda x: x[1])[
                        0
                    ]
                else:
                    # Default to "string" if there are no calculated probabilities
                    most_probable_dtype = "string"

                confirmed_data_types[col] = most_probable_dtype
                action = (
                    "defaulted to"
                    if most_probable_dtype == "string"
                    and total_analyzed == 0
                    and empty_count > 0
                    else "accepted as"
                )
                print(
                    f"{col}: Data type {action} {confirmed_data_types[col].upper()}.\n"
                )
                break

            print(
                "Invalid response. Please enter a valid data type code or press Enter to accept the inferred type.\n"
            )

    return confirmed_data_types


def infer_data_types_and_counts(column_data, col_name, in_date_format):
    """
    Efficiently analyzes column data to count occurrences of each data type using batch operations,
    adapted to work with a Series of consolidated unique values for a single column.

    Parameters:
    - column_data: A pandas Series containing the unique data of a single column.
    - col_name: The name of the column being analyzed. (Currently unused but kept for potential future use)
    - in_date_format: The date format to use for parsing dates.

    Returns:
    - A dictionary with data types as keys and their counts as values.
    """
    data_type_counts = defaultdict(int)
    if in_date_format:
        date_converted = pd.to_datetime(
            column_data, format=in_date_format, errors="coerce"
        )
        data_type_counts["date"] = date_converted.notna().sum()

    booleans_detected = column_data.str.lower().isin(
        ["true", "false", "t", "f", "1", "0"]
    )
    data_type_counts["boolean"] = booleans_detected.sum()

    numeric_candidates = column_data[~booleans_detected]
    if not numeric_candidates.empty:
        numeric_converted = pd.to_numeric(numeric_candidates, errors="coerce")
        floats_detected = numeric_converted.notna() & (numeric_converted % 1 != 0)
        data_type_counts["float"] = floats_detected.sum()
        data_type_counts["integer"] = (
            numeric_converted.notna().sum() - floats_detected.sum()
        )
    else:
        data_type_counts["float"] = 0
        data_type_counts["integer"] = 0

    # Determine if any types have been identified, adjust string count accordingly
    identified_counts = sum(data_type_counts.values())
    total_values = len(column_data)
    data_type_counts["string"] = (
        total_values - identified_counts if identified_counts < total_values else 0
    )

    # Ensure default to string if no other data types identified
    if not any(data_type_counts.values()):
        data_type_counts["string"] = total_values

    return dict(data_type_counts)


def consolidate_and_analyze_columns(files):
    """
    Consolidates unique values for each column across all provided CSV files, processes each
    consolidated column in parallel to determine data type distributions, and calculates
    probabilities for each data type in each column.

    Parameters:
    - files: List of file paths to the CSV files.

    Returns:
    - A dictionary with column names as keys, each mapping to another dictionary where keys
      are data types and values are the probabilities of each data type for the column.
    - A dictionary with the total counts of values analyzed for each column.
    """
    # Consolidate unique values for each column across all files into pandas Series
    consolidated_columns = defaultdict(list)
    empty_counts = defaultdict(int)
    for file_path in tqdm(files, desc="Reading Files"):
        df = pd.read_csv(file_path, dtype=str, low_memory=False)
        for col in df.columns:
            empty_counts[col] += df[col].isna().sum()
            consolidated_columns[col].extend(df[col].dropna().unique())

    for col, values in consolidated_columns.items():
        consolidated_columns[col] = (
            pd.Series(values).drop_duplicates().reset_index(drop=True)
        )

    # Process each consolidated column in parallel to determine data type distributions
    column_data_types = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(infer_data_types_and_counts, data, col, in_date_format): col
            for col, data in consolidated_columns.items()
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Analyzing columns"
        ):
            col = futures[future]
            column_data_types[col] = future.result()

    # Accumulate data type counts and calculate probabilities
    total_counts_accumulator = {
        col: sum(counts.values()) for col, counts in column_data_types.items()
    }
    probabilities = {
        col: {
            dtype: (
                (count / total_counts_accumulator[col])
                if total_counts_accumulator[col] > 0
                else 0
            )
            for dtype, count in counts.items()
        }
        for col, counts in column_data_types.items()
    }

    return probabilities, total_counts_accumulator, empty_counts


def get_data_types_and_clean():
    """
    Orchestrates the entire process including user interaction for data type decisions and data sanitization,
    now using consolidated column values for data type probability processing.
    """
    # Set global formats based on user input
    global confirmed_data_types
    confirmed_data_types["conversion_issues"] = "string"
    get_user_numeric_format_choice()
    get_user_date_format_choices()

    # Gather all CSV file paths in the input folder
    files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(".csv")
    ]

    # Use the consolidated function for accumulating data type counts and calculating probabilities
    accumulated_probabilities, total_counts_accumulator, empty_values_count = (
        consolidate_and_analyze_columns(files)
    )

    # Extract column order from the first file
    first_file_path = files[0]
    df = pd.read_csv(first_file_path, dtype=str, nrows=0)
    column_order = df.columns.tolist()

    # Display the calculated probabilities to the user for confirmation
    display_data_type_probabilities(
        accumulated_probabilities,
        total_counts_accumulator,
        empty_values_count,
        column_order,
    )

    # Interaction for data type edits based on probabilities remains unchanged
    edit_choice = (
        input("Do you want to edit any of these data types? (Y/N): ").strip().lower()
    )

    if edit_choice == "y":
        confirmed_data_types = confirm_or_edit_data_types(
            accumulated_probabilities,
            column_order,
            empty_values_count,
            total_counts_accumulator,
        )
    else:
        for col, probs in accumulated_probabilities.items():
            confirmed_data_types[col] = max(probs, key=probs.get)

    # Ask if the user wants to sanitize and clean the data based on confirmed data types
    handle_sanitization_choice(files)


def display_data_type_probabilities(
    accumulated_probabilities,
    total_counts_accumulator,
    empty_values_count,
    column_order,
):
    """
    Displays the probabilities of data types for each column, including a count of empty fields,
    with columns ordered as they appear in the input files.

    Parameters:
    - accumulated_probabilities: Dict of probabilities for each data type in each column.
    - total_counts_accumulator: Dict of total counts of values analyzed for each column.
    - empty_values_count: Dict of empty value counts for each column.
    - column_order: List of column names in the order they appear in the input files.
    """
    # Prepare the headers and the rows for the table
    headers = [
        "Column",
        "Total Analyzed",
        "Empty Fields",
        "Date",
        "Boolean",
        "Float",
        "Integer",
        "String",
    ]
    rows = []

    for col in column_order:
        if col in accumulated_probabilities:  # Check if the column is in the results
            probs = accumulated_probabilities[col]
            total_analyzed = total_counts_accumulator[col]
            empty_count = empty_values_count[col]
            row = [
                col.center(len(col)),
                str(total_analyzed).center(len("Total Analyzed")),
                str(empty_count).center(len("Empty Fields")),
            ]

            for dtype in ["date", "boolean", "float", "integer", "string"]:
                prob = probs.get(dtype, 0) * 100  # Convert to percentage
                cell_value = f"{prob:.2f}%" if prob > 0 else "-"
                if (
                    dtype == "string" and empty_count > 0 and total_analyzed == 0
                ):  # Check if column is completely empty
                    cell_value = " default"
                row.append(cell_value.center(len(dtype)))

            rows.append(row)

    # Print the table
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def handle_sanitization_choice(files):
    """
    Handles user choice for sanitizing data, updating the script to process based on confirmed data types.
    """
    global confirmed_data_types  # Ensure we're modifying the global variable
    print(
        "\nDo you want to sanitize the data based on the confirmed data types? This can alter your dataset by removing rows that do not match the specified data types. (Y/N/Help):"
    )
    choice = input().strip().lower()
    if choice == "help":
        # Explain the sanitization process and implications
        print(
            "Sanitization will check each row in your CSV files against the confirmed data types. Rows with data that does not match these types will be excluded from the processed dataset. This step can help ensure data quality but may also lead to data loss if many rows do not conform to the expected types."
        )
        choice = (
            input("Do you want to proceed with sanitization? (Y/N): ").strip().lower()
        )

    if choice == "y":
        sanitize_and_save_data(files)
    elif choice == "n":
        print(
            "Skipping data sanitization. All columns will be set to use dtype string to avoid errors."
        )
        print(
            "Note: If you wish to perform operations that require specific data types (numeric, boolean, or dates), you will need to sanitize your data."
        )
        # Set all columns to dtype string
        for col in confirmed_data_types.keys():
            confirmed_data_types[col] = "string"


def sanitize_and_save_data(files):
    """
    Sanitizes data for each file based on confirmed data types and saves the cleaned data.
    """

    def output_summary(summaries):
        total_files = len(summaries)
        total_clean_files = 0
        total_unclean_files = 0
        total_rows_processed = 0
        total_dirty_num_rows = 0
        total_dirty_bool_rows = 0
        total_dirty_date_rows = 0

        for summary in summaries:
            # debugging output
            # print("\n-------------------------------------------------")
            # print(f"File processed: {summary['file_path']}")
            # print(f"Clean rows: {summary['clean_rows_count']}")
            # print(f"Rows with issues: {summary['rows_with_issues_count']}")
            # print("-------------------------------------------------")

            # Increment counters based on summary info
            if summary["rows_with_issues_count"] > 0:
                total_unclean_files += 1
                total_dirty_num_rows += summary["dirty_num_rows"]
                total_dirty_bool_rows += summary["dirty_bool_rows"]
                total_dirty_date_rows += summary["dirty_date_rows"]
            else:
                total_clean_files += 1

            total_rows_processed += summary["total_rows"]

        # Calculate overall success rate
        dirty_rows_total = (
            total_dirty_num_rows + total_dirty_bool_rows + total_dirty_date_rows
        )
        clean_rows_total = total_rows_processed - dirty_rows_total
        success_rate_files = (
            (total_clean_files / total_files) * 100 if total_files else 0
        )
        success_rate_rows = (
            (clean_rows_total / total_rows_processed) * 100
            if total_rows_processed
            else 0
        )
        failure_rate_dirty_rows = (
            (dirty_rows_total / total_rows_processed) * 100
            if total_rows_processed
            else 0
        )

        # Output overall summary
        data = [
            [
                "Files processed",
                total_files,
                "",
                "Rows processed",
                total_rows_processed,
                "",
                "Dirty num rows",
                total_dirty_num_rows,
            ],
            [
                "Clean files",
                total_clean_files,
                "",
                "Dirty rows",
                dirty_rows_total,
                "",
                "Dirty date rows",
                total_dirty_date_rows,
            ],
            [
                "Dirty files",
                total_unclean_files,
                "",
                "Clean rows",
                clean_rows_total,
                "",
                "Dirty bool rows",
                total_dirty_bool_rows,
            ],
            # Success rates row
            [
                "Success rate (files)",
                f"{success_rate_files:.2f}%",
                "",
                "Success rate (rows)",
                f"{success_rate_rows:.2f}%",
                "",
                "Failure rate (rows)",
                f"{failure_rate_dirty_rows:.2f}%",
            ],
        ]

        headers = [
            "Metric (Files)",
            "Value",
            " ",
            "Metric (Rows)",
            "Value",
            " ",
            "Specific Issues",
            "Value",
        ]
        print("\nSummary of sanitization process:")
        print(tabulate(data, headers=headers, tablefmt="grid"))

    summaries = []
    with ProcessPoolExecutor() as executor:
        # Create a list to hold the futures
        futures = []
        for file in files:
            future = executor.submit(
                process_and_sanitize_data,
                file,
                output_folder,
                confirmed_data_types,
                in_date_format,
                out_date_format,
                numeric_format,
                boolean_format,
            )
            futures.append(future)

        # Use tqdm to display progress
        for future in tqdm(
            as_completed(futures), total=len(files), desc="Processing Files"
        ):
            # Results can be processed here if needed
            try:
                # Getting the result also checks for any exceptions raised during execution
                summary_info = future.result()
                summaries.append(summary_info)
            except Exception as e:
                print(f"An error occurred: {e}")

    # Output summary after all files are processed
    output_summary(summaries)


def process_and_sanitize_data(
    file_path,
    output_folder,
    confirmed_data_types,
    in_date_format,
    out_date_format,
    numeric_format,
    boolean_format,
):
    """
    Process and sanitize a single CSV file based on specified data types and formats,
    tracking conversion issues in a dedicated column. Returns a summary of the process.

    Parameters:
    - file_path: Path to the input CSV file.
    - output_folder: Path where the sanitized file and any issues file should be saved.
    - confirmed_data_types: Dict specifying the data type for each column.
    - in_date_format: Date format for parsing dates.
    - out_date_format: Date format for outputting dates.
    - numeric_format: Dict specifying numeric format for parsing numbers.
    - boolean_format: Bool format for outputting booleans.
    """
    # print(f"Starting processing and sanitization of {file_path}.")
    df = pd.read_csv(file_path, dtype=str, low_memory=False)
    # print("Initial CSV read completed.")

    # Initialize a column to track conversion issues at the row level
    df["conversion_issues"] = np.nan

    summary = {
        "file_path": file_path,
        "total_rows": len(df),
        "rows_with_issues_count": 0,
        "clean_rows_count": 0,
        "dirty_num_rows": 0,
        "dirty_bool_rows": 0,
        "dirty_date_rows": 0,
    }

    # Iterate through each column to process based on its data type
    for col, dtype in confirmed_data_types.items():
        # print(f"Processing column: {col} with data type: {dtype}")

        # Skip processing if column not found in DataFrame
        if col not in df.columns:
            tqdm.write(f"Column {col} not found in the file. Skipping...")
            continue

        # Initialize a Series for individual column's conversion issues
        conversion_issues = pd.Series(index=df.index, dtype="object")

        # Process column based on its data type
        if dtype == "date":
            df[col], conversion_issues = preprocess_date(
                df[col], in_date_format, out_date_format, conversion_issues
            )
            # Increment dirty_date_rows for each conversion issue in this column
            summary["dirty_date_rows"] += conversion_issues.notna().sum()
        elif dtype in ["float", "integer"]:
            df[col], conversion_issues = preprocess_numeric_series(
                df[col], numeric_format, conversion_issues
            )
            # Increment dirty_num_rows for each conversion issue in this column
            summary["dirty_num_rows"] += conversion_issues.notna().sum()
        elif dtype == "boolean":
            df[col], conversion_issues = preprocess_boolean_series(
                df[col], boolean_format, conversion_issues
            )
            # Increment dirty_bool_rows for each conversion issue in this column
            summary["dirty_bool_rows"] += conversion_issues.notna().sum()

        # Assuming 'conversion_issues' is filled with non-null values for rows with issues
        # Update the 'conversion_issues' column in the dataframe
        if "conversion_issues" in df.columns:
            df["conversion_issues"] = df["conversion_issues"].combine_first(
                conversion_issues
            )
        else:
            df["conversion_issues"] = conversion_issues

    # print("Completed processing for all columns.")

    # Identify rows with any conversion issues and drop rows where 'conversion_issues' is NaN before saving issues file
    rows_with_issues = df[
        df["conversion_issues"].notna() & (df["conversion_issues"] != "")
    ]

    # Save the cleaned data, including only rows without identified conversion issues (NaN in 'conversion_issues')
    clean_rows = df[pd.isna(df["conversion_issues"])].drop(
        columns=["conversion_issues"]
    )
    clean_file_path = os.path.join(
        output_folder, os.path.basename(file_path).replace(".csv", "_clean.csv")
    )
    clean_rows.to_csv(clean_file_path, index=False)
    # print(f"Clean data saved to {clean_file_path}.")

    summary["rows_with_issues_count"] = len(rows_with_issues)
    summary["clean_rows_count"] = len(clean_rows)

    # Add a check to ensure the dirty rows counters add up to rows with issues count
    if summary["rows_with_issues_count"] != (
        summary["dirty_num_rows"]
        + summary["dirty_bool_rows"]
        + summary["dirty_date_rows"]
    ) or (
        summary["clean_rows_count"]
        != summary["total_rows"] - summary["rows_with_issues_count"]
    ):
        print(
            "Error: The sum of dirty rows does not match the total rows with issues count."
        )
    else:
        # print("Counts verified: The sum of dirty rows matches the total rows with issues count.")
        pass

    # Save rows with conversion issues, if any
    if not rows_with_issues.empty:
        issues_file_path = os.path.join(
            output_folder, os.path.basename(file_path).replace(".csv", "_issues.csv")
        )
        # Include only the columns of interest for issues file; you might want to keep 'conversion_issues'
        rows_with_issues.to_csv(issues_file_path, index=False)
        # print(f"Rows with issues saved to {issues_file_path}.")

    return summary


# add check for custom number format and add multiple formats (spaces as thou seps, no thou seps, etc)
def get_user_numeric_format_choice():
    global numeric_format  # Declare the use of the global variable within the function

    print("\nSelect the numeric format for your CSV:")
    print("1: Dot as decimal separator (e.g., 1,000.00)")
    print("2: Comma as decimal separator (e.g., 1.000,00)")
    print("c: Custom format")
    choice = input("Your choice (1/2/c): ").strip()

    if choice == "1":
        numeric_format = {"decimal": ".", "thousands": ","}
    elif choice == "2":
        numeric_format = {"decimal": ",", "thousands": "."}
    elif choice == "c":
        decimal = input("Enter your decimal separator (e.g., '.', ','): ").strip()
        thousands = input(
            "Enter your thousands separator (if any, else leave blank): "
        ).strip()
        numeric_format = {"decimal": decimal, "thousands": thousands}
    else:
        print("Defaulting to dot as decimal separator.")
        numeric_format = {"decimal": ".", "thousands": ","}
        # The numeric_format is already initialized with the default values, so no need to update it unless the user chooses a different format.


# add check for custom date format
def get_user_date_format_choices():
    """Prompts the user for input and output date formats with explanations and a help option."""
    global in_date_format, out_date_format
    print(
        "\nYour CSV files might contain dates. Specifying the correct date format ensures accurate processing."
    )
    print("If you're unsure about the date formats, type 'help' for examples.")

    has_dates = input("Do CSV files contain date values? (Y/N/Help): ").strip().lower()
    if has_dates == "help":
        # Provide examples and implications of date format selection
        print(
            "\nHelp: Choosing 'Y' allows you to specify the date format present in your CSV files."
        )
        has_dates = input("Do CSV files contain date values? (Y/N): ").strip().lower()

    if has_dates == "y":
        date_formats = {
            "1": "%d/%m/%Y",
            "2": "%m/%d/%Y",
            "3": "%Y-%m-%d",
            "4": "%d-%m-%Y",
            "5": "%Y/%m/%d",
        }
        print("\nSelect the input date format for your CSV:")
        for key, fmt in date_formats.items():
            print(f"{key}: {fmt}")
        print("c: Custom format")
        choice = input("Your choice: ").strip().lower()
        in_date_format = (
            date_formats.get(choice)
            if choice in date_formats
            else (
                input("Enter your custom input date format: ").strip()
                if choice == "c"
                else None
            )
        )

        print("\nSelect the output date format for your CSV (if different from input):")
        for key, fmt in date_formats.items():
            print(f"{key}: {fmt}")
        print("c: Custom format, s: Same as input")
        choice = input("Your choice: ").strip().lower()
        out_date_format = (
            in_date_format
            if choice == "s"
            else (
                date_formats.get(choice)
                if choice in date_formats
                else (
                    input("Enter your custom output date format: ").strip()
                    if choice == "c"
                    else None
                )
            )
        )
    else:
        in_date_format = out_date_format = None


def preprocess_numeric_series(column, numeric_format, conversion_issues):
    """
    Preprocess a numeric series based on the user-selected numeric format,
    handling negative numbers, removing thousands separators, and dealing with decimal separators.
    Flags unsuccessful conversions and updates the conversion_issues Series accordingly.

    Parameters:
    - column: Pandas Series to be processed.
    - numeric_format: Dict with 'thousands' and 'decimal' separators.
    - conversion_issues: Pandas Series to flag rows with conversion issues.

    Returns:
    - The Series with numeric conversions applied.
    - Updated conversion_issues Series with flags for unsuccessful conversions.
    """
    # Replace negative sign with a marker to avoid interference during formatting
    column = column.str.replace("-", "NEG", regex=False)

    # Remove any spaces
    column = column.str.replace(" ", "", regex=False)

    # Attempt to clean and convert the series based on the provided numeric format
    if numeric_format["thousands"]:
        column = column.str.replace(numeric_format["thousands"], "", regex=False)
    if numeric_format["decimal"] and numeric_format["decimal"] != ".":
        column = column.str.replace(numeric_format["decimal"], ".", regex=False)

    # Convert back the 'NEG' marker to '-'
    column = column.str.replace("NEG", "-", regex=False)

    converted = pd.to_numeric(column, errors="coerce")

    # Flag rows with conversion issues
    conversion_issues.loc[converted.isna() & column.notna()] = "num"

    # Apply formatting to remove unnecessary decimal points for whole numbers
    formatted = converted.apply(
        lambda x: (
            format(x, ".0f")
            if pd.notnull(x) and isinstance(x, float) and x.is_integer()
            else x
        )
    )

    # Return converted data where possible; otherwise, keep the original data
    column = column.where(converted.isna(), formatted.astype(str))

    return column, conversion_issues


def preprocess_date(column, in_date_format, out_date_format, conversion_issues):
    """
    Convert date strings from a pandas Series to the specified input and output date formats,
    marking conversion failures with a specific flag.

    Parameters:
    - column: Series to be processed.
    - in_date_format: Expected format of the input dates.
    - out_date_format: Desired format of the output dates.
    - conversion_issues: Series to flag conversion issues.

    Returns:
    - The Series with date conversions applied where possible.
    - Updated conversion_issues Series with flags for unsuccessful conversions.
    """
    original_data = column.copy()
    converted = pd.to_datetime(column, format=in_date_format, errors="coerce")

    # Flag rows with conversion issues
    conversion_issues.loc[converted.isna() & column.notna()] = "date"

    # Apply the output date format to successfully converted dates
    if out_date_format:
        column = converted.dt.strftime(out_date_format).where(
            ~converted.isna(), original_data
        )

    return column, conversion_issues


# add boolean input check and ask for standardization of all values to a certain format
def preprocess_boolean_series(column, boolean_format, conversion_issues):
    """
    Preprocess a series for boolean values, considering 'true', 'false', 't', 'f', '1', and '0'.
    Flags inconsistencies and ignores empty fields, updating conversion issues accordingly.

    Parameters:
    - column: Pandas Series containing potential boolean values.
    - conversion_issues: Series to flag conversion issues.

    Returns:
    - The Series with boolean conversions applied where possible.
    - Updated conversion_issues Series with flags for unsuccessful conversions.
    """
    original_data = column.copy()
    bool_mapping = {
        "true": True,
        "t": True,
        "1": True,
        "false": False,
        "f": False,
        "0": False,
    }
    converted = column.str.lower().map(bool_mapping)

    # Identify rows that couldn't be converted (excluding empty values)
    conversion_issues.loc[converted.isna() & column.notna()] = "bool"

    # For successfully converted rows, replace the original data with converted boolean values
    column = converted.where(~converted.isna(), original_data)

    return column, conversion_issues


def int_dict():
    return defaultdict(int)


if __name__ == "__main__":
    main()
