import sys

try:
    from .cli import cli
except ImportError as e:
    print(f"Error: Missing dependencies. Please install with 'poetry install'. Details: {e}")
    sys.exit(1)


def main():
    """Main entry point for the D&D transcriber package."""
    try:
        cli()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
