import pandas as pd
df = pd.read_csv("data.csv",index_col=0) 

vari = df['Minutes Played'].var()
for columns in ['Wind', 'Temp', 'Humidity', 'Outlook']:
    u = df[columns].unique()
    s = 0.0
    for x in u:
        subset_df = df[df[columns]==x] 
        t_1 =subset_df['Minutes Played'].var() 
        t_2 = 1.0*len(subset_df)/len(df)*t_1
        print(x, t_1, t_2)
        s = s + t_2
        #s = s + subset_df['Minutes Played'].var()
        #print(s)
    print(columns, s, vari-s)
    print("*"*20)
