#!/bin/sh

echo "WARNING: This will clear all documents and vector database!"
echo "Are you sure you want to proceed? (Y/N)"
read -r response

if [ "$response" != "Y" ] && [ "$response" != "y" ]; then
  echo "Script aborted."
  exit 1
fi

rm -rf /app/docs/*
rm -rf /app/storage/docs/*
echo "CLEARED: UPLOADED DOCUMENTS"

rm -rf /app/db/*
rm -rf /app/storage/db/*
echo "CLEARED: VECTOR DATABASE"

echo "SYSTEM RESET COMPLETE!"