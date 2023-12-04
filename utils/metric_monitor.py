from collections import defaultdict


class MetricMonitor:
    """
    The `MetricMonitor` class is a utility class designed to monitor and compute metrics during training or evaluation
    of machine learning models. It allows for tracking multiple metrics, accumulating values, and computing their
    averages over time.

    Parameters:
    - float_precision: int, optional (default=4)
      - The number of decimal places to display for the computed metric averages.

    Attributes:
    - float_precision: int
      - The precision for displaying computed metric averages.
    - metrics: defaultdict
      - A dictionary that stores metrics as key-value pairs, where each metric includes values for 'val' (accumulated sum),
       'count' (number of updates), and 'avg' (average value).

    Methods:
    - reset():
      - Resets all tracked metrics, clearing accumulated values and counts.

    - update(metric_name, val):
      - Updates a specific metric with a new value 'val'.
      - Accumulates the value, increments the count, and computes the new average.

    - __str__():
      - Converts the metrics and their computed averages to a formatted string for display.

    Description:
    - The `MetricMonitor` class is a valuable tool for monitoring the progress of model training or evaluation.
    - It allows you to keep track of multiple metrics simultaneously, making it suitable for tasks with various
      performance measures.
    - Metrics are accumulated and averaged over time, providing insights into the model's performance during each
      training epoch or evaluation step.
    - The `float_precision` parameter controls the precision of displayed averages, ensuring readability.
    - You can reset the metrics at any time using the `reset` method and update them with new values using the
      `update` method.
    - The class provides a convenient string representation of metrics, making it easy to include in logs or print
      during training and evaluation.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
