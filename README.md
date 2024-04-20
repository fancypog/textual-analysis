# Textual Analysis

This repository contains code for textual analysis of mutual funds' prospectuses.

## Overview

This repository encompasses various scripts for textual analysis, covering the following main functionalities:

1. **Sample Filtering**: Scripts to filter a sample of text files based on specific criteria, such as the presence of series ID and ticker symbols.
2. **Data Cleaning**: Scripts to ensure data consistency and uniformity by cleaning various columns in the DataFrame `final_path`.
3. **PIS Section Isolation**: Scripts to extract the Principal Investment Strategies (PIS) section from a collection of text files.

## Features

### Sample Filtering

This script filters a sample of text files based on specific criteria, such as series ID and ticker symbol presence:

- **Set Working Directory**: Establishes the working directory.
- **Retrieve File List**: Obtains file paths for all .txt files.
- **Filter Sample**: Extracts relevant information and filters data based on criteria.
- **Construct DataFrame**: Creates a DataFrame (`final_path`) with the filtered data.

### Data Cleaning

This script ensures data consistency and uniformity by cleaning various columns in the DataFrame `final_path`:

- **Clean `form_type` Column**: Removes tab characters ('\t') and joins elements.
- **Clean `company_name` Column**: Removes tab characters ('\t') and joins elements.
- **Clean `CIK` Column**: Removes tab characters ('\t') and joins elements.
- **Clean `fdate` Column**: Removes tab characters ('\t') and joins elements.
- **Clean `rdate` Column**: Removes tab characters ('\t') and joins elements.
- **Drop Duplicates and Add Columns**: Drops duplicate entries, resets the index, and adds new columns to `file_final_path`.

### PIS Section Isolation

This script extracts the Principal Investment Strategies (PIS) section from a collection of text files:

- **Define Possible Expressions**: Lists of expressions indicate the start and end of the PIS section.
- **Generate Combinations**: Forms possible combinations of start and end expressions.
- **Extract PIS Section**: Utilizes a function `extract_letter` to extract the PIS section text.
- **Iterate Through Files**: Processes each file in the `file_final_path` DataFrame.
- **Save Results to CSV**: Saves the extracted PIS sections to a CSV file ('PIS_full.csv').


