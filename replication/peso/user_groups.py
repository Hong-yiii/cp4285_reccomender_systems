"""Label each block's test users as new, dormant or continuing from PESO block files.

Follows the paper's Table 3 groups (arXiv 2510.25093v3, §5.2): new users first appear
in the test block; dormant users appeared in an earlier block, skipped the preceding
block and returned. Membership uses the released block files, not raw timestamps.

    python replication/peso/user_groups.py local/peso/data/Instruments_cl_ori \
        Instruments_cl_ori runs/peso/instruments-user-groups.json
"""

import json
import sys
from collections import Counter
from pathlib import Path


def groups(blocks):
    present = [set(block) for block in blocks]
    labels = {}
    for b in range(1, len(blocks)):
        labels[str(b)] = {
            user: "continuing"
            if user in present[b - 1]
            else "dormant"
            if any(user in earlier for earlier in present[: b - 1])
            else "new"
            for user, data in blocks[b].items()
            if len(data["current_block"]) >= 3
        }
    return labels


def main(root, name, output, num_blocks=5):
    blocks = [
        json.loads((Path(root) / f"{name}.all.block_{b}.json").read_text())
        for b in range(num_blocks)
    ]
    labels = groups(blocks)
    with Path(output).open("x") as f:  # create-only, like every project artifact
        json.dump(labels, f)
    for block, users in labels.items():
        print(f"block {block}: {dict(Counter(users.values()))}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
