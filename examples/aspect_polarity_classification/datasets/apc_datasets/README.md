The Chinese datasets has binary polarity labeled as 0 (Negative), 1 (Positive)
The multilingual dataset is the sum of other datasets and convert the Chinese polarities from {0, 1} to {0, 2}, in order
to fit the triple sentiment categories of English datasets.

However, when construct your custom dataset, you should label the sentiment polarities in [0, N-1] (i.e., N types of
polarities)