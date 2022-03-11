#!/bin/bash

search_dir=$1
for entry in "$search_dir"/*
do
  echo "$entry"
  python3 client_har.py -f "$entry" &
done