import pandas as pd 
import torch
from chronos import BaseChronosPipeline

def chronos_test(filename, time, target):
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
        device_map="cpu",  # use "cpu" for CPU inference
        torch_dtype=torch.bfloat16,
    )

    df = pd.read_csv(
        "static/files/" + filename
    )

    prediction_length = 12
    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    # quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
    # mean is an fp32 tensor with shape [batch_size, prediction_length]
    quantiles, mean = pipeline.predict_quantiles(
        context=torch.tensor(df[target]),
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

    # Convert the time column to datetime
    time_df = pd.to_datetime(df[time])

    # Calculate the time difference between consecutive times
    time_differences = time_df.diff().dropna()

    # Assume the pattern continues with the most common difference
    most_common_difference = time_differences.value_counts().idxmax()

    # Generate the next time
    last_time = time_df.iloc[-1]
    
    next_time = last_time + most_common_difference

    # Define how many next times you want (e.g., generate the next 5 times)
    next_times = [next_time + i * most_common_difference for i in range(0, prediction_length)]

    data = {time:next_times, "low": low.tolist(), "median": median.tolist(), "high": high.tolist()}

    output_df = pd.DataFrame(data)

    output_df.to_csv('static/files/output.csv', index=False)
    
    return output_df