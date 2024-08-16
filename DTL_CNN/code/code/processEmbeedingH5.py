import h5py
import pandas as pd


def processh5(filename):
    # filename = "/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code_bert.h5"

    cui_dict={}

    with h5py.File(filename, "r") as f:
        n=0
        v_len=len(f[list(f.keys())[0]][()][0])
        # print(v_len)
        # print(len(list(f.keys())))
        for key in list(f.keys()):
            cui_dict[key]=list(f[key])[0]
            n+=1
            # if n%10000==0:
            #     print(n)

    cui_df=pd.DataFrame.from_dict(cui_dict, orient='index', columns=['V'+ str(i) for i in range(v_len)])
    
    return cui_df

