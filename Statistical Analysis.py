import pandas as pd

def Important_Collumn_Averages(file_path):
    #Pega todas as colunas do arquivo CSV e tira a sua média somente das colunas que possuem informação numérica
    print(file_path)

    Data_Frame = pd.read_csv(file_path)
    headers = Data_Frame.columns.tolist()

    for header in headers:
        if not pd.api.types.is_numeric_dtype(Data_Frame[header]):  # Check if column is numeric
            #print(header)
            continue
        else:
            media_coluna = Data_Frame.loc[:,header].mean()
            print(f"{header}: {media_coluna}")

    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")



#------------------------------------------------------------------------------------------------------------- Code starts Here-----------------------------------------------------------------------------------------------

#O vídeo possui 175 frames, ou 7 segundos, ou 7 000 ms. Logo, o "Total frame time(ms)" deve chegar em aproximadamente 7 000 / 175 = 40ms

Important_Collumn_Averages('Encode_Fixed_QP_Log_0.csv')
Important_Collumn_Averages('Encode_ROI_Log_0.csv')
        