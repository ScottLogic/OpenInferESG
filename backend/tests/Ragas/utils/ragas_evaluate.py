#!/usr/bin/env python3
"""
RAGAS Evaluation Command Line Tool
---------------------------------
A command-line tool to evaluate question-answering systems using RAGAS metrics.
"""
import asyncio
from modules.ragas_cli import main

if __name__ == "__main__":
    asyncio.run(main())
