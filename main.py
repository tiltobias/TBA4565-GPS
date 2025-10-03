import georinex as gr

print("georinex version:", gr.__version__)

data = gr.load("RINEX.nav").to_dataframe()
# print(data)
print(data.columns)

print(data.xs("G03", level="sv").iloc[1])