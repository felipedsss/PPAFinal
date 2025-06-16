import pandas as pd
import os

diretorio = "empresas"

if not os.path.exists(diretorio):
    os.makedirs(diretorio)
    print(f"Diretório '{diretorio}' criado.")
else:
    print(f"Diretório '{diretorio}' já existe.")

fullDf = pd.read_csv('stock_details_5_years.csv')
fullDf = fullDf.drop(columns=['Volume','Dividends','Stock Splits'])
print(fullDf['Company'].to_list())

company_list = fullDf['Company'].unique()
# Salva um CSV para cada empresa dentro do diretório 'empresas'
for company in company_list:
    companyDf = fullDf[fullDf['Company'] == company]
    companyDf = companyDf.drop(columns=['Company'])
    filename = os.path.join("empresas", f"{company}.csv")
    companyDf.to_csv(filename, index=False)
    print(f"Saved stock details for {company} to {filename}")