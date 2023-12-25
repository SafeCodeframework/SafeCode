import json
import os
import re
from pycparser import c_parser, preprocess_file
import ast

parser = c_parser.CParser()


def remove_comment(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def check_syntax(code, parser):
    try:
        preprocessed_code = preprocess_file(code, cpp_args='-E')
        ast = parser.parse(preprocessed_code)
    except Exception as e:
        print(e)
        print(preprocessed_code)
        return False
    return True



