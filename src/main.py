import argparse

parser = argparse.ArgumentParser(description='A simple example of linear regression.')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

args = parser.parse_args()

if __name__ == "__main__":
    if args.verbose:
        print("Verbose Hello!")
    print("Hello World!")