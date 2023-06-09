# ML4QS

## How to use `merge.py`

### Parameters:

`-a`: Merge activities separately

`-m`: Merge existed activities into raw.csv

### Example
If there are not existed activity files, merge activities separately:

`python merge.py -a`

Then merge existed activities into raw.csv:

`python merge.py -m`

Above two commands can be merged into one:

`python merge.py -a -m`

## How to use `visualization.py`

### Parameters:

`-f`: The path of file to be visualized

### Example
`python visualization.py -f datasets/intermediate/raw.csv`