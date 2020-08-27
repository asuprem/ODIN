import numpy as np
import sqlite3, io

def adapt_array(arr):
    # [RODRIGO]: Pushing numpy array to sqlite (conversion).
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    # [RODRIGO]: Pushing numpy array to sqlite (conversion).
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)