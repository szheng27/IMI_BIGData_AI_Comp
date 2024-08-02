#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V1 Created in Jan 2024

Team 37
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Ernest (Khashayar) Namdar
"""

# Importing the required libraries ############################################
"""
Note: We could accelerate the operations using cuDF and cuML from RAPIDS
Nonetheless, team members had installation issues depending on the platform they used
"""

import sqlite3
import pandas as pd


# Create a connection to the SQLite database
conn = sqlite3.connect('../results/Scotiabank.sqlite')



# Function to create tables in the database
def create_tables(conn):
    # SQL to create kyc table
    kyc_table = '''
    CREATE TABLE IF NOT EXISTS kyc (
        cust_id TEXT PRIMARY KEY,
        Name TEXT,
        Gender TEXT,
        Occupation TEXT,
        Age INTEGER,
        Tenure INTEGER,
        label INTEGER
    );
    '''


    # SQL to create cash_trxns table
    cash_trxns_table = '''
    CREATE TABLE IF NOT EXISTS cash_trxns (
        cash_trxn_id TEXT PRIMARY KEY,
        cust_id TEXT,
        amount REAL,
        type TEXT,
        FOREIGN KEY (cust_id) REFERENCES kyc (cust_id)
    );
    '''

    # SQL to create emt_trxns table
    emt_trxns_table = '''
    CREATE TABLE IF NOT EXISTS emt_trxns (
        emt_trxn_id TEXT PRIMARY KEY,
        id_sender TEXT,
        id_receiver TEXT,
        name_sender TEXT,
        name_receiver TEXT,
        emt_message TEXT,
        emt_value REAL,
        FOREIGN KEY (id_sender) REFERENCES kyc (cust_id),
        FOREIGN KEY (id_receiver) REFERENCES kyc (cust_id)
    );
    '''

    # SQL to create wire_trxns table
    wire_trxns_table = '''
    CREATE TABLE IF NOT EXISTS wire_trxns (
        wire_trxn_id TEXT PRIMARY KEY,
        id_sender TEXT,
        id_receiver TEXT,
        name_sender TEXT,
        name_receiver TEXT,
        wire_value REAL,
        country_sender TEXT,
        country_receiver TEXT,
        FOREIGN KEY (id_sender) REFERENCES kyc (cust_id),
        FOREIGN KEY (id_receiver) REFERENCES kyc (cust_id)
    );
    '''

    # Execute the SQL statements
    with conn:
        conn.execute(kyc_table)
        conn.execute(cash_trxns_table)
        conn.execute(emt_trxns_table)
        conn.execute(wire_trxns_table)

# Function to load CSV data into the database and rename columns
def load_csv_to_table(conn, csv_path, table_name, column_renames):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    # Rename columns
    df.rename(columns=column_renames, inplace=True)
    # Write the DataFrame to the SQL table
    df.to_sql(table_name, conn, if_exists='append', index=False)

# Create the tables
create_tables(conn)

# Load the CSV files into the database
# Assuming the CSV files are named 'kyc.csv', 'cash_trxns.csv', 'emt_trxns.csv', 'wire_trxns.csv'
# and are located in the /mnt/data directory.
# The renaming dictionary is used to match CSV columns with the database schema


load_csv_to_table(conn, '../data/kyc.csv', 'kyc', {})
load_csv_to_table(conn, '../data/cash_trxns.csv', 'cash_trxns', {
    'trxn_id': 'cash_trxn_id'
})

load_csv_to_table(conn, '../data/emt_trxns.csv', 'emt_trxns', {
    'trxn_id': 'emt_trxn_id',
    'id sender': 'id_sender',
    'id receiver': 'id_receiver',
    'name sender': 'name_sender',
    'name receiver': 'name_receiver',
    'emt message': 'emt_message',
    'emt value': 'emt_value'

})

load_csv_to_table(conn, '../data/wire_trxns.csv', 'wire_trxns', {
    'trxn_id': 'wire_trxn_id',
    'id sender': 'id_sender',
    'id receiver': 'id_receiver',
    'name sender': 'name_sender',
    'name receiver': 'name_receiver',
    'wire value': 'wire_value',
    'country sender': 'country_sender',
    'country receiver': 'country_receiver'

})

# Commit changes and close the connection
conn.commit()
conn.close()
