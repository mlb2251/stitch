import json
import os
from textwrap import dedent

import neurosym as ns

root = "data/python"

os.makedirs(root, exist_ok=True)


def create_python_test(name, *pythons):
    s_exp = [ns.python_to_s_exp(code) for code in pythons]
    with open(os.path.join(root, f"{name}.json"), "w") as f:
        json.dump(s_exp, f, indent=2)


create_python_test(
    "multi-arg-function",
    "func1(x, y, z, a, b, c) + 10",
    "func2(x, y, z, a, b, c)",
    "2 + 10",
    "3 + 10",
    "4 + 10",
)

create_python_test(
    "front-of-sequence",
    dedent(
        """
        function(x, y, z)
        2 + 3 + 4
        distraction
        distraction2 + 2 + 3 + 4
        """
    ),
    dedent(
        """
        function(x, y, z2)
        2 + 3 + 4
        """
    ),
    dedent(
        """
        function(x, y, z3)
        2 + 3 + 4
        distraction3
        distraction4 + 2 + 3 + 4
        """
    ),
    dedent(
        """
        function(x, y, z4)
        2 + 3 + 4
        distraction4
        """
    ),
    dedent(
        """
        function(x, y, z5)
        2 + 3 + 4
        distraction
        8 + 273 + 1
        """
    ),
)

create_python_test(
    "back-of-sequence",
    dedent(
        """
        a
        b
        c
        distraction
        distraction2 + 2 + 3 + 4
        function(x, y, z)
        2 + 3 + 4
        """
    ),
    dedent(
        """
        distraction4 + 2 + 3 + 4
        function(x, y, z2)
        2 + 3 + 4
        """
    ),
    dedent(
        """
        distraction4
        function(x, y, z3)
        2 + 3 + 4
        """
    ),
    dedent(
        """
        u
        v
        distraction
        distraction5 + 2 + 3 + 4
        function(x, y, z4) 
        2 + 3 + 4
        """
    ),
    dedent(
        """
        distraction6
        distraction7 + 2 + 3 + 4
        function(x, y, z5)
        2 + 3 + 4
        """
    ),
)
