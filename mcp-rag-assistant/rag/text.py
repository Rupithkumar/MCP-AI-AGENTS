from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it its
    of on or such than that the their then there these they this to was we were what when
    where which who why will with you your about should would could may might not no""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_VOWELS = frozenset("aeiou")


def stem(word: str) -> str:
    w = word
    # 1) plurals
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith(("sses", "xes", "ches", "shes", "zes")):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
 
    for suffix, min_base in (("ing", 3), ("ed", 4)): 
        if w.endswith(suffix) and len(w) - len(suffix) >= min_base:
            base = w[: -len(suffix)]
            if any(c in _VOWELS for c in base):
                w = base
                if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz": 
                    w = w[:-1]
                break
   
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
