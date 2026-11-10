#!/usr/bin/env python3
"""A minimal hello world program."""


def greet(name: str = "World") -> str:
    return f"Hello, {name}!"


def main() -> None:
    print(greet())


if __name__ == "__main__":
    main()
