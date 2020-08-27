import codecs
import json
import sys


def std_flush(*args, **kwargs):
    """Write a space-separated tuple of arguments to the standard output and flush its buffers."""
    print(" ".join(map(str, args)))
    sys.stdout.flush()


def load_json(json_):
    return json.load(codecs.open(json_, encoding='utf-8'))

