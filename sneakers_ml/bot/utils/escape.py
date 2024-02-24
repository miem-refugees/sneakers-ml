import re


def escape(text):
    return re.sub(r"[_*[\]()~>#\+\-=|{}.!]", lambda x: "\\" + x.group(), text)
