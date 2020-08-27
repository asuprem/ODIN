# -*- coding: utf-8 -*-
#!/usr/bin/python
# Code from https://github.com/SmileYuhao/concept-drift with minor changes (some function names to fit interface)
import mlep.drift_detector.LabeledDriftDetector.LabeledDriftDetector as LabeledDriftDetector

class ADWIN(LabeledDriftDetector.LabeledDriftDetector):
    """ Implements the ADWIN drift detection method.

    The ADWIn (Adaptive Windowing) drift detector is based the concept of adaptive sliding windows. It is based on paper ADWIN paper (Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive windowing."). This implementation is adapted from https://github.com/SmileYuhao/concept-drift. 

    Attributes:
        min_instances: Minimum number of instances required for delivering a detection prediction
        drift_level: Alarm level for drift detection. DDM Paper specified 3.0 for out-of-control and 2.0 for warning

    Methods:
        __init__: Initialize the detector
        reset: Reset the Drift detector
        detect: Perform drift detection

    """
    #def __init__(self, delta=0.002, max_buckets=5, min_clock=32, min_win_len=10, min_sub_win_len=5):
    def __init__(self, delta=0.002, max_buckets=5, min_clock=32, min_win_len=10, min_sub_win_len=5):
        """
        Initialize the ADWIN Drift Detector
        
        Initialize the ADWIN Drift detector. Default parameters are provided as well

        Args:
            delta: FLOAT. Confidence value
            max_buckets: INT. Max number of buckets within one bucket row
            min_clock: INT. Min number of new data for starting to reduce window and detect change
            min_win_len: INT. Min window length for starting to reduce window and detect change
            min_sub_win_len: INT. Min sub-window length, which is split from whole window
        """
        from math import log, fabs, sqrt

        self.log = log
        self.fabs=fabs
        self.sqrt=sqrt
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_clock = min_clock
        self.min_win_len = min_win_len
        self.min_sub_win_len = min_sub_win_len

        # Time is used for comparison with min_clock parameter
        self.time = 0
        # Current window length
        self.window_len = 0
        # Sum of all values in the window
        self.window_sum = 0.0
        # Variance of all values in the window
        self.window_variance = 0.0

        # Count of bucket row within window
        self.bucket_row_count = 0
        # Init bucket list
        self.bucket_row_list = AdwinRowBucketList(self.max_buckets)



    
    def reset(self,):
        """ 
        Reset the drift detector.

        Resets the ADWIN Drift detector. This should be called after `detect()` returns True to reset internal parameters.

        """
        
        # Time is used for comparison with min_clock parameter
        self.time = 0
        # Current window length
        self.window_len = 0
        # Sum of all values in the window
        self.window_sum = 0.0
        # Variance of all values in the window
        self.window_variance = 0.0

        # Count of bucket row within window
        self.bucket_row_count = 0
        # Init bucket list
        self.bucket_row_list = AdwinRowBucketList(self.max_buckets)

    def detect(self,error):
        """ 
        Perform Drift Detection
        
        `detect` performs Drift Detection according to the method specified in the ADWIN Paper.

        Args:
            Error: INT. 1 if classification was incorrect; 0 if classification was correct
        
        Returns:
            Bool: True is there is drift; False if there is no drift

        """

        self.time += 1

        # Insert the new element
        self.__insert_element(error)

        # Reduce window
        return self.__reduce_window()
        
        
    def __insert_element(self, value):
        """
        Add new element to ADWIN.

        Create a new bucket, and insert it into bucket row which is the head of bucket row list.
        Meanwhile, this bucket row maybe compressed if reaches the maximum number of buckets.

        Args:
            value: New data value from the stream
            
        """
        # Insert the new bucket
        self.bucket_row_list.head.insert_bucket(value, 0)

        # Calculate the incremental variance
        incremental_variance = 0
        if self.window_len > 0:
            mean = self.window_sum / self.window_len
            incremental_variance = self.window_len * pow(value - mean, 2) / (self.window_len + 1)

        # Update statistic value
        self.window_len += 1
        self.window_variance += incremental_variance
        self.window_sum += value

        # Compress buckets if necessary
        self.__compress_bucket_row()

    def __compress_bucket_row(self):
        """
        Bucker compression.

        If ADWIN reaches maximum number of buckets, then merge two buckets within one row into a next bucket row.

        """
        bucket_row = self.bucket_row_list.head
        bucket_row_level = 0
        while bucket_row is not None:
            # Merge buckets if row is full
            if bucket_row.bucket_count == self.max_buckets + 1:
                next_bucket_row = bucket_row.next_bucket_row
                if next_bucket_row is None:
                    # Add new bucket row and move to it
                    self.bucket_row_list.add_to_tail()
                    next_bucket_row = bucket_row.next_bucket_row
                    self.bucket_row_count += 1

                # Calculate number of bucket which will be compressed into next bucket row
                n_1 = pow(2, bucket_row_level)
                n_2 = pow(2, bucket_row_level)
                # Mean
                mean_1 = bucket_row.bucket_sum[0] / n_1
                mean_2 = bucket_row.bucket_sum[1] / n_2
                # Total
                next_total = bucket_row.bucket_sum[0] + bucket_row.bucket_sum[1]
                # Variance
                external_variance = n_1 * n_2 * pow(mean_1 - mean_2, 2) / (n_1 + n_2)
                next_variance = bucket_row.bucket_variance[0] + bucket_row.bucket_variance[1] + external_variance

                # Insert into next bucket row, meanwhile remove two original buckets
                next_bucket_row.insert_bucket(next_total, next_variance)

                # Compress those tow buckets
                bucket_row.compress_bucket(2)

                # Stop if the number of bucket within one row, which does not exceed limited
                if next_bucket_row.bucket_count <= self.max_buckets:
                    break
            else:
                break

            # Move to next bucket row
            bucket_row = bucket_row.next_bucket_row
            bucket_row_level += 1

    def __reduce_window(self):
        """
        Detect a change from last of each bucket row, reduce the window if there is a concept drift.

        Return:
            Boolean: Whether there is concept drift
        """
        is_changed = False
        if self.time % self.min_clock == 0 and self.window_len > self.min_win_len:
            is_reduced_width = True
            while is_reduced_width:
                is_reduced_width = False
                is_exit = False
                n_0, n_1 = 0, self.window_len
                sum_0, sum_1 = 0, self.window_sum

                # Start building sub windows from the tail of window (old bucket row)
                bucket_row = self.bucket_row_list.tail
                i = self.bucket_row_count
                while (not is_exit) and (bucket_row is not None):
                    for bucket_num in range(bucket_row.bucket_count):
                        # Iteration of last bucket row, or last bucket in one row
                        if i == 0 and bucket_num == bucket_row.bucket_count - 1:
                            is_exit = True
                            break

                        # Grow sub window 0, while reduce sub window 1
                        n_0 += pow(2, i)
                        n_1 -= pow(2, i)
                        sum_0 += bucket_row.bucket_sum[bucket_num]
                        sum_1 -= bucket_row.bucket_sum[bucket_num]
                        diff_value = (sum_0 / n_0) - (sum_1 / n_1)

                        # Minimum sub window length is matching
                        if n_0 > self.min_sub_win_len + 1 and n_1 > self.min_sub_win_len + 1:
                            # Remove oldest bucket if there is a concept drift
                            if self.__reduce_expression(n_0, n_1, diff_value):
                                is_reduced_width, is_changed = True, True
                                if self.window_len > 0:
                                    n_0 -= self.__delete_element()
                                    is_exit = True
                                    break

                    # Move to previous bucket row
                    bucket_row = bucket_row.previous_bucket_row
                    i -= 1
        return is_changed

    def __reduce_expression(self, n_0, n_1, diff_value):
        """
        Calculate epsilon cut value.

        Args:
            n_0: number of elements in sub window 0
            n_1: number of elements in sub window 1
            diff_value: difference of mean values of both sub windows

        Return:
            Boolean: true if difference of mean values is higher than epsilon_cut
        """
        # Harmonic mean of n0 and n1
        m = 1 / (n_0 - self.min_sub_win_len + 1) + 1 / (n_1 - self.min_sub_win_len + 1)
        d = self.log(2 * self.log(self.window_len) / self.delta)
        variance_window = self.window_variance / self.window_len
        epsilon_cut = self.sqrt(2 * m * variance_window * d) + 2 / 3 * m * d
        return self.fabs(diff_value) > epsilon_cut

    def __delete_element(self):
        """
        Remove a bucket from tail of bucket row.
        :return: Number of elements to be deleted
        """
        bucket_row = self.bucket_row_list.tail
        deleted_number = pow(2, self.bucket_row_count)
        self.window_len -= deleted_number
        self.window_sum -= bucket_row.bucket_sum[0]

        deleted_bucket_mean = bucket_row.bucket_sum[0] / deleted_number
        inc_variance = bucket_row.bucket_variance[0] + deleted_number * self.window_len * pow(
            deleted_bucket_mean - self.window_sum / self.window_len, 2
        ) / (deleted_number + self.window_len)

        self.window_variance -= inc_variance

        # Delete bucket from bucket row
        bucket_row.compress_bucket(1)
        # If bucket row is empty, remove it from the tail of window
        if bucket_row.bucket_count == 0:
            self.bucket_row_list.remove_from_tail()
            self.bucket_row_count -= 1

        return deleted_number


class AdwinRowBucketList:
    def __init__(self, max_buckets=5):
        """
        Args:
            max_buckets: Max number of bucket in each bucket row
        """
        self.max_buckets = max_buckets

        self.count = 0
        self.head = None
        self.tail = None
        self.__add_to_head()

    def __add_to_head(self):
        """
        Init bucket row list.

        """
        self.head = AdwinBucketRow(self.max_buckets, next_bucket_row=self.head)
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def add_to_tail(self):
        """
        Add the bucket row at the end of the window.

        """
        self.tail = AdwinBucketRow(self.max_buckets, previous_bucket_row=self.tail)
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def remove_from_tail(self):
        """
        Remove the last bucket row in the window.

        """
        self.tail = self.tail.previous_bucket_row
        if self.tail is None:
            self.head = None
        else:
            self.tail.next_bucket_row = None
        self.count -= 1


class AdwinBucketRow:
    def __init__(self, max_buckets=5, next_bucket_row=None, previous_bucket_row=None):
        """

        Args:
            max_buckets: Max bucket with one row
            next_bucket_row: Following bucket row
            previous_bucket_row: Previous bucket row
        """
        self.max_buckets = max_buckets
        import numpy as np
        self.np = np

        # Current count of buckets in this bucket row
        self.bucket_count = 0

        self.next_bucket_row = next_bucket_row
        # Set previous bucket row connect to this bucket row
        if next_bucket_row is not None:
            next_bucket_row.previous_bucket_row = self

        self.previous_bucket_row = previous_bucket_row
        # Set next bucket connect to this bucket
        if previous_bucket_row is not None:
            previous_bucket_row.next_bucket_row = self

        # Init statistic number for each bucket
        self.bucket_sum = self.np.zeros(self.max_buckets + 1)
        self.bucket_variance = self.np.zeros(self.max_buckets + 1)

    def insert_bucket(self, value, variance):
        """
        Insert a new bucket at the end.

        """
        self.bucket_sum[self.bucket_count] = value
        self.bucket_variance[self.bucket_count] = variance
        self.bucket_count += 1

    def compress_bucket(self, number_deleted):
        """
        Remove the oldest buckets.

        """
        delete_index = self.max_buckets - number_deleted + 1
        self.bucket_sum[:delete_index] = self.bucket_sum[number_deleted:]
        self.bucket_sum[delete_index:] = self.np.zeros(number_deleted)

        self.bucket_variance[:delete_index] = self.bucket_variance[number_deleted:]
        self.bucket_variance[delete_index:] = self.np.zeros(number_deleted)

        self.bucket_count -= number_deleted    