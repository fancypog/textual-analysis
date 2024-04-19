# =============================================================================
# 1. DATA CLEANING AND PIS EXTRACTION
# =============================================================================

import os
import re
import pandas as pd
import itertools
import numpy as np

# =============================================================================
# Step 1 filter sample based on whether the document has series id and ticker:


# Set working directory
new_directory = 'D:/701 technical'
os.chdir(new_directory)

# Define a function to get the file path
def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path): # files contain: (1) all sub-directory; and (2) all files within the directory
        for file in files:
            if file.lower().endswith('txt'): #identify if the file is txt form
                file_list.append(os.path.join(root,file)) #add to the file list
    return file_list

file_list = get_file_list(os.getcwd())

# Filter
form_list = []
file_date = []
r_date = []
company_list = []
CIK_list = []
series_list = []
fund_list = []
class_list = []
content_list = []
filename_list = []
path_list = []

for path in file_list:
    f = open(path, 'r', errors='ignore')
    buff = f.read()
    buff = buff.replace('\n', '').upper()

    # Identify which form:N-CSR or N-CSRS
    form_type = re.findall(r'CONFORMED SUBMISSION TYPE:(.*?)PUBLIC DOCUMENT COUNT:', buff)
    # CONFORMED PERIOD OF REPORT: The actual dates of holdings, which are used to merge with holding data
    # FILED AS OF DATE: The dates when a NCSR form was filed, which are used to test the market reactions.
    # For example as below:
    # CONFORMED PERIOD OF REPORT: 20201231
    # FILED AS OF DATE: 20210302
    rdate = re.findall(r'CONFORMED PERIOD OF REPORT:(.*?)FILED AS OF DATE', buff)
    filedate = re.findall(r'FILED AS OF DATE:(.*?)DATE AS OF CHANGE', buff)
    company_name = re.findall(r'COMPANY CONFORMED NAME:(.*?)CENTRAL INDEX KEY', buff)
    CIK = re.findall(r'CENTRAL INDEX KEY:(.*?)IRS NUMBER:', buff)
    filename = os.path.basename(path)

    w1 = '<SERIES>'
    w2 = '</SERIES>'
    pat = re.compile(w1 + '(.*?)' + w2, re.S)
    result = pat.findall(buff)


    for i in result:
        pos = i.find('<SERIES-ID')
        series_id = re.findall(r'<SERIES-ID>(.*?)<SERIES-NAME>',i)
        fund_name = re.findall(r'<SERIES-NAME>(.*?)<CLASS-CONTRACT>',i)
        class_ticker = re.findall(r'<CLASS-CONTRACT-TICKER-SYMBOL>(.*?)</CLASS-CONTRACT>',i)
        if series_id !=[] and class_ticker !=[]:
            form_list.append(form_type)
            file_date.append(filedate)
            r_date.append(rdate)
            company_list.append(company_name)
            CIK_list.append(CIK)
            series_list.append(series_id)
            fund_list.append(fund_name)
            class_list.append(class_ticker)
            path_list.append(path)
            filename_list.append(filename)

#Fund level data
final_path = pd.DataFrame({'file': path_list, 'filename': filename_list, 'form_type': form_list,'CIK': CIK_list, 'company_name':company_list,'fdate': file_date, 'rdate': r_date,
                         'seriesid': series_list, 'fund_name': fund_list, 'ticker':class_list})

# =============================================================================
# Step 2 Clean content

# clean form_type
form_list = []
for mid in final_path['form_type']:
    new_form = ''.join(mid)
    new_form = new_form.replace('\t', '')
    form_list.append(new_form)

final_path['form_type'] = form_list
final_path = final_path.dropna().reset_index(drop=True)


# clean company_name
company_list = []
for mid in final_path['company_name']:
    new_name = ''.join(mid)
    new_name = new_name.replace('\t', '')
    company_list.append(new_name)

final_path['company_name'] = company_list
final_path = final_path.dropna().reset_index(drop=True)


# clean CIK
CIK_list = []
for mid in final_path['CIK']:
    new_CIK = ''.join(mid)
    new_CIK = new_CIK.replace('\t', '')
    CIK_list.append(new_CIK)

final_path['CIK'] = CIK_list
final_path = final_path.dropna().reset_index(drop=True)


# clean file_date
fdate_list = []
for date in final_path['fdate']:
    new_date = ''.join(date)
    new_date = new_date.replace('\t', '')
    fdate_list.append(new_date)

final_path['fdate'] = fdate_list
final_path = final_path.dropna().reset_index(drop=True)


# clean r_date
rdate_list = []
for date in final_path['rdate']:
    new_date = ''.join(date)
    new_date = new_date.replace('\t', '')
    rdate_list.append(new_date)

final_path['rdate'] = rdate_list
final_path = final_path.dropna().reset_index(drop=True)
#Originally, we have 47 documents in our sample. But here we have 101 lists. 
#The reason is that a document may contain several funds (seriesid).


file_final_path = final_path.drop_duplicates('file').reset_index(drop = True)


file_final_path['letter_content'] = 0
file_final_path['letter_dummy'] = 0


# =============================================================================
# Step 3 Get the PIS section

# List possible expressions at the start and the end of PIS section
start_words = ['PRINCIPAL INVESTMENT STRATEGIES', 'PRINCIPAL INVESTMENT STRATEGY', 'PRINCIPAL STRATEGIES', 'PRINCIPAL STRATEGY']
end_words = ['PRINCIPAL RISKS', 'PRINCIPAL INVESTMENT RISKS', 'INVESTMENT STRATEGY RISK', 'MAIN RISKS', 'PRINCIPAL RISK']

# Form possible combinations of the start and end of the PIS section
combinations = list(itertools.product(start_words, end_words))

# Extract text between all possible combinations of start words and end words
# If text, dummy = 1; If none, dummy = 0

def extract_letter(start, end, content):
    pat = re.compile(start + '(.*?)' + end)
    letter = pat.findall(content)
    if letter != []:
        dummy = 1
    else:
        dummy = 0
        
    return letter, dummy

for i in range(len(file_final_path)):
    path = file_final_path.loc[i, 'file']
    f = open(path, 'r', errors='ignore')
    buff = f.read()
    buff = buff.replace('\n', ' ').upper()
    
    content = re.sub(r'\<.*?\>', ' ', buff)
    content = re.sub(r'[^a-zA-Z0-9]', ' ', content)
    
    found = False
    extracted_content = None
    for startword, endword in combinations:
        letter, dummy = extract_letter(startword, endword, content)
        if dummy != 0:
            extracted_content = letter[0]
            file_final_path.loc[i, 'letter_dummy'] = dummy
            found = True
            break  # Exit the loop if a valid combination is found
    
    if found:
        file_final_path.loc[i, 'letter_content'] = extracted_content
        continue  # Move to the next file
        
    # If no valid combination is found, try another combination
    for startword, endword in combinations:
        letter, dummy = extract_letter(startword, endword, content)
        if dummy != 0:
            file_final_path.loc[i, 'letter_content'] = letter[0]
            file_final_path.loc[i, 'letter_dummy'] = dummy
            break  # Exit the loop if a valid combination is found
            

# Save the results to a csv
file_final_path.to_csv('D:/701 technical/PIS_full.csv', index=False)


# =============================================================================
# 2. DATA MERGE
# =============================================================================
import pandas as pd
import statsmodels.api as sm

# =============================================================================
# Prepare and merge data
# =============================================================================
# Step 1 Generate quant search code

# data.csv is a file with fund names with quant/non-quant, AI/non-AI dummy variables from GPT-4/ word-matching
df = pd.read_csv('data.csv')

# Quant no. 31607   AI 2202
# One Company contains several funds with the same prospectus, so funds share one dummy variable under the same firm
value_vars = [str(i) for i in range(23)]
quant_sq = df.melt(id_vars=['quant_dummy'], value_vars=value_vars, value_name='search_code')
mask1 = quant_sq['search_code'].apply(lambda x: (isinstance(x, int) or isinstance(x, float) or (isinstance(x, str) and len(x) <= 4)))
quant_sq2 = quant_sq[~mask1]
quant_sq2.dropna()
# Keep those with dummy "1"
quant_sq2.to_csv('quant_sq2.csv')

# Do the same to generate quant dummy variables
ai_sq = df.melt(id_vars=['AI_dummy'], value_vars=value_vars, value_name='search_code')
mask2 = ai_sq['search_code'].apply(lambda x: (isinstance(x, int) or isinstance(x, float) or (isinstance(x, str) and len(x) <= 4)))
ai_sq2 = ai_sq[~mask2]
ai_sq2.dropna()

ai_sq2.to_csv('ai_sq2.csv')

# Step 2 Merge data
# Use the search code to merge dummies with returns of the fund
quant_sc = pd.read_csv('search_code_quant.csv')
ai_sc = pd.read_csv('search_code_AI.csv')
ret = pd.read_csv('ret.csv')
# Merge the quant and AI dummy data
ret['quant_dummy'] = ret['search_code'].isin(search_code['isQuant']).astype(int)
ret['AI_dummy'] = ret['search_code'].isin(search_code['isAI']).astype(int)
# Output the merged data
ret.to_csv('merged_ret.csv')

# =============================================================================
# 3. Regression
# =============================================================================
# Prepare your data (More variables to be added)
df = pd.read_csv('merged_ret.csv')
df = df.drop(df[df['mret'].apply(lambda x: isinstance(x, str))].index)

# Assuming you have your dependent variable 'y' and independent variables 'x1', 'x2', 'x3' in your DataFrame
y = df['mret']
X = df[['mtna','mnav','quant_dummy']]
Z = df[['mtna','mnav','AI_dummy']]
# Convert y to numpy array and specify data type as float64
y = np.asarray(y).astype(np.float64)

# Convert X to numpy array
X = np.asarray(X)

# Add a constant column to the independent variables
X = sm.add_constant(X)
# Fit the linear regression model
model = sm.OLS(y, X)
results = model.fit()

# Obtain the statistical summary
# The first regression including quant funds
print(results.summary())

# The second regression including AI funds
Z = sm.add_constant(Z)
model2 = sm.OLS(y,Z)
results2 = model2.fit()
print(results2.summary)



