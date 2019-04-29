#!/bin/bash
rm models/*/results-*
git add models/
git commit -m "models update"
git push
