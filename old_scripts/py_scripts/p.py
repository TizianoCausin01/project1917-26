import sys

arg = sys.argv[1]              # "1 2 3"
parsed_list = list(map(int, arg.split()))

print(parsed_list)
