import re

def parse_trba_output(path="trba_output.txt"):
    results = []

    pattern = re.compile(r'(\S+)\s+([^\s]+)\s+([0-9.]+)')

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                img, jp, conf = match.groups()
                results.append((img, jp, float(conf)))

    return results
