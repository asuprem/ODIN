import datetime
import time

def readable_time(format="%M:%S"):
    """Return a string representing the current time, controlled by an explicit format string.

    format -- [str] Format string.
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime(format)

def ms_to_readable(ms, format="%Y-%m-%d %H:%M:%S"):
    """Return a string representing the specified timestamp in milliseconds, controlled by an
    explicit format string.

    ms -- [int] Timestamp in milliseconds to be formatted.
    format -- [str] Format string.
    """
    return datetime.datetime.fromtimestamp(ms/1000).strftime(format)

def time_to_id(ms=None, lval=5):
    """Return a string obtained from mapping digits of the specified timestamp into characters.

    ms -- [int] Timestamp in milliseconds to be transformed into a string. If it is None, use
    current timestamp.
    """
    DICTA = {str(idx): item for (idx, item) in enumerate("abcdefghij")}
    if ms is None:
        ms = time.time()
    ms_str = ("%."+str(lval)+"f")%ms
    ms_str = ms_str[:-(lval+1)]+ms_str[-lval:]
    return ''.join([DICTA[item] for item in ms_str])