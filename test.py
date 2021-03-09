# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-09
# @FilePath     : /LuWu/test.py
# @Desc         :
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()


def foo(args):
    print(args.x * args.y)


def bar(args):
    print("((%s))" % args.z)


parser_foo = subparsers.add_parser("foo")
parser_foo.add_argument("y", type=float)
parser_foo.add_argument("x", type=float)
parser_foo.set_defaults(cmd="web-server")

parser_bar = subparsers.add_parser("bar")
parser_bar.add_argument("z", help="zzzz")
parser_bar.set_defaults(cmd="object-detection")

args = parser.parse_args()
print(args)

if args.cmd == "web-server":
    print(f"{args.x}+{args.y}={args.x+args.y}")
else:
    print(f"(({args.z}))")
