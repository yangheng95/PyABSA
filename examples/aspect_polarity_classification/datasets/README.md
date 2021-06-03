The Chinese datasets has binary polarity labeled as 0 (Negative), 1 (Positive)
The multilingual dataset is the sum of other datasets and convert the Chinese polarities to -1, 1

However, you should label the sentiment polarities in [0, N] (N+1 types of polarities)