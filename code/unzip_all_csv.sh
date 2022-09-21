#!/bin/bash

search_dir=$1
for entry in "$search_dir"/*
do
  echo "$entry"
  gzip -d "$entry"
done