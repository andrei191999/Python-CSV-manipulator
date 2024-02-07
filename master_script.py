# TO ADD
# for mandatory field check, create additional csv for rows with multiple missing values.


import os
import re
import unicodedata
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Global variable for sender name
sender_name = ""
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    global sender_name, timestamp
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
    data_types = get_data_types(input_folder)

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
            check_individual_and_combo_duplicates(
                input_folder, output_folder, data_types
            )
        elif choice == "2":
            count_empty_values(input_folder, output_folder, data_types)
        elif choice == "3":
            add_or_update_column(input_folder, output_folder, data_types)
        elif choice == "4":
            strip_or_remove_spaces_from_columns(input_folder, output_folder, data_types)
        elif choice == "5":
            split_and_order_csvs(input_folder, output_folder, data_types)
        elif choice == "6":
            process_mandatory_fields(input_folder, output_folder, data_types)
        elif choice == "7":
            replace_substring_in_column(input_folder, output_folder, data_types)
        elif choice == "8":
            count_value_occurrences(input_folder, output_folder, data_types)
        elif choice == "9":
            format_numbers_or_dates_in_column(input_folder, output_folder, data_types)
        elif choice == "10":
            rename_column_header(input_folder, output_folder, data_types)
        elif choice == "0":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")


def rename_column_header(input_folder, output_folder, data_types):
    column_prompt = "Enter the column number/name to rename: "
    current_column_header, _ = get_column_names(
        input_folder, None, column_prompt, return_single_col=True
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
def format_numbers_or_dates_in_column(input_folder, output_folder, data_types):
    column_prompt = "Enter the column number/name to format numbers or dates: "
    column_name, existing_columns = get_column_names(
        input_folder,
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
                df = read_csv_with_dtypes(file_path, data_types)
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


def split_and_order_csvs(input_folder, output_folder, data_types):
    global sender_name, timestamp
    column_prompt = "Enter the column number/name to order by: "
    order_column, existing_columns = get_column_names(
        input_folder,
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
                df = read_csv_with_dtypes(file_path, data_types)
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


def count_value_occurrences(input_folder, output_folder, data_types):
    global timestamp, sender_name

    # Allow selection of multiple columns
    column_prompt = "Enter column names/numbers to count value/substring occurrences (separated by commas): "
    column_names, _ = get_column_names(
        input_folder,
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

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)

            try:
                df = read_csv_with_dtypes(file_path, data_types)

                for col in column_names:
                    if col in df.columns:
                        if is_substring:
                            mask = df[col].apply(lambda x: value_to_search in str(x))
                        else:
                            mask = df[col] == value_to_search

                        rows_with_searched_value = df[mask]

                        # Update the counter
                        occurrence_counters[col] += rows_with_searched_value.shape[0]

                        if not rows_with_searched_value.empty:
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
                            output_file_path = os.path.join(
                                output_folder, f"{sanitized_filename}"
                            )
                            rows_with_searched_value.to_csv(
                                output_file_path, index=False
                            )
                            print(
                                f"Rows with '{value_to_search}' in column '{col}' saved to {output_file_path}"
                            )

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Print occurrence counts
    for col, count in occurrence_counters.items():
        print(f"Occurrences in '{col}': {count}")

    if all(count == 0 for count in occurrence_counters.values()):
        print(f"No rows found with '{value_to_search}' in the selected columns.")


def replace_substring_in_column(input_folder, output_folder, data_types):
    column_prompt = "Enter the column number/name in which to replace the substring: "
    column_name, existing_columns = get_column_names(
        input_folder,
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
                df = read_csv_with_dtypes(file_path, data_types)

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


def process_mandatory_fields(input_folder, output_folder, data_types):

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
        input_folder,
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
                df = read_csv_with_dtypes(file_path, data_types)

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


def strip_or_remove_spaces_from_columns(input_folder, output_folder, data_types):
    column_prompt = "Enter column names/numbers to modify (separated by commas), or press enter to process all columns: "

    column_names, _ = get_column_names(
        input_folder,
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
                df = read_csv_with_dtypes(file_path, data_types)

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


def add_or_update_column(input_folder, output_folder, data_types):
    column_prompt = "Enter the column number to update, '-1' to add a new column, or column name directly: "
    column_name, _ = get_column_names(
        input_folder,
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
                df = read_csv_with_dtypes(input_file_path, data_types)
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


def parse_user_input_for_columns_and_groups(input_folder):
    column_prompt = "Enter columns and groups to check for duplicates (e.g., '1,5,doc_date,[1,3,doc_date]'): "
    column_names, _ = get_column_names(
        input_folder,
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
                input_folder,
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


def check_individual_and_combo_duplicates(input_folder, output_folder, data_types):
    global sender_name, timestamp

    # Get column names or groups to check for duplicates
    column_names = parse_user_input_for_columns_and_groups(input_folder)

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
                df = read_csv_with_dtypes(file_path, data_types)
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


def count_empty_values(input_folder, output_folder, data_types):
    global sender_name, timestamp
    placeholders = ["", " ", "nan", "-", "_", ".", "na", "null"]

    column_prompt = "Enter column names/numbers to count empty values (separated by commas), or press enter to check all columns: "
    column_names, existing_columns = get_column_names(
        input_folder,
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
                df = read_csv_with_dtypes(file_path, data_types)

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
    input_folder,
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


""" Sanitize a string to be used as a valid filename. Removes or replaces characters that are not
allowed in filenames, trims whitespace, and avoids names that could be problematic for a file system. """


def sanitize_filename(name):
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


def add_quotes_to_column(df, column_name):
    if column_name in df.columns:
        df.loc[:, column_name] = '"' + df[column_name].astype(str) + '"'
    return df


def read_csv_with_dtypes(file_path, data_types):
    try:
        converters = {
            col: preprocess_float
            for col, dtype in data_types.items()
            if dtype == "float64" or dtype == "int64"
        }
        data_types = {
            col: dtype
            for col, dtype in data_types.items()
            if dtype != "float64" and dtype != "int64"
        }
        df = pd.read_csv(file_path, dtype=data_types, converters=converters)
        # Print final data types
        # print("Final Data Types after processing:")
        # print(df.dtypes)
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()


def get_data_types(input_folder):
    # Function to map dtype to short code
    def dtype_to_code(dtype):
        mapping = {
            "object": "s",
            "int64": "i",
            "float64": "f",
            "bool": "b",
            # Add other mappings if needed
        }
        return mapping.get(str(dtype), "s")

    # Attempt to load the first CSV file in the directory and display inferred data types
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_csv(file_path, nrows=5)  # Read first few rows
                # Adjust data types for columns with all empty values
                for col in df.columns:
                    if df[col].isnull().all():
                        df[col] = df[col].astype(str)
                print("\nPython's inferred data types for the first CSV file:")
                for col in df.columns:
                    print(f"  {col}: {dtype_to_code(df[col].dtype)}")
                break
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                return {}

    # Warnings about data type implications
    print("\nWarning about Data Type Choices:")
    print("  - Choosing 'bool' will camel case 'True'/'False' values.")
    print("  - Choosing 'float' will remove spaces and add decimals, even for .0.")
    print(
        "  - It's usually best to use 'string' unless you specifically need number or boolean formatting."
    )

    # Ask user whether to set all columns to strings
    choice = (
        input(
            "\nSet all columns to strings instead of inferred types? (yes [Y]/no [N]): "
        )
        .strip()
        .lower()
    )
    default_to_str = choice == "y"

    # Reload DataFrame with chosen data types
    df = pd.read_csv(file_path, nrows=5, dtype=str if default_to_str else None)

    # Ask if the user wants to manually edit data types
    edit_choice = (
        input(
            "Do you want to manually edit any of the column data types? (yes [Y]/no [N]): "
        )
        .strip()
        .lower()
    )
    if edit_choice != "y":
        return {col: "str" if default_to_str else df[col].dtype for col in df.columns}

    # Prompt user for data types
    print("\nSpecify data types for each column (current type shown in brackets):")
    print("  s: string, i: integer, f: float, b: boolean")
    print("Hit enter to confirm or enter a new type.")

    data_type_options = {
        "s": "str",
        "i": "int64",
        "f": "float64",
        "b": "bool",
        # Add other options if needed
    }

    data_types = {}
    for col in df.columns:
        current_type = dtype_to_code(df[col].dtype)
        user_input = (
            input(f"{col} [{current_type if not default_to_str else 's'}]: ")
            .strip()
            .lower()
        )
        data_types[col] = data_type_options.get(
            user_input, "str" if default_to_str else df[col].dtype
        )

    return data_types


def preprocess_float(value):
    """Remove spaces and convert to float."""
    try:
        return float(str(value).replace(" ", ""))
    except ValueError:
        return None  # or return pd.NA if you want to use pandas' NA type


if __name__ == "__main__":
    main()
