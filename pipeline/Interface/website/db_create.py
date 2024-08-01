#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 Created in Dec 2023

Team X
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Ernest (Khashayar) Namdar
"""

import sqlite3
import pandas as pd

# Path to the CSV files
kyc_file = 'website/data/kyc_with_open_sanctions.csv'
cash_trxns_file = 'website/data/cash_trxns.csv'
emt_trxns_file = 'website/data/emt_trxns.csv'
wire_trxns_file = 'website/data/wire_trxns.csv'
found_criminals_file = 'website/data/Interface_t3_criminals.csv'

# Database file
db_file = 'Scotia.db'

# Create a new SQLite database
conn = sqlite3.connect(db_file)

# Function to load a CSV file into the database
def load_csv_to_db(csv_file, table_name, conn):
    df = pd.read_csv(csv_file)
    df.to_sql(table_name, conn, if_exists='replace', index=False)

# Load each CSV file into a separate table
load_csv_to_db(kyc_file, 'kyc', conn)
load_csv_to_db(cash_trxns_file, 'cash_trxns', conn)
load_csv_to_db(emt_trxns_file, 'emt_trxns', conn)
load_csv_to_db(wire_trxns_file, 'wire_trxns', conn)
load_csv_to_db(found_criminals_file,'found_criminals',conn)

# Close the database connection
conn.close()
