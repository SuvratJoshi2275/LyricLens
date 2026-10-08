"""CLI entrypoint for training.

Kept separate from trainer.py so users can run the expected:

    python train.py
"""

from trainer import main


if __name__ == "__main__":
    main()
