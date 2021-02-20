import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--echo", help="echo the string you use here")

args = parser.parse_args()

if args.echo:
    print(args.echo)

else:
    print("None passed")
