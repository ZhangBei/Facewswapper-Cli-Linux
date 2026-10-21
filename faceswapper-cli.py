#!/usr/bin/env python3
"""Headless command-line entry point.

    python faceswapper-cli.py --models ./models swap \
        --source face.jpg --target target.jpg --output out.png
"""

import sys

if __name__ == "__main__":
    from faceswapper.cli import main
    sys.exit(main())
