# ASC-analyzer

The ASC Analyzer extracts Argument Structure Constructions (ASCs) from raw English texts and computes indices related to ASC usage.


## Installation
To ensure stability and compatibility, we recommend installing dependencies in the following order:

1. Install `spaCy`:
   ```bash
   pip install spacy
   ```

2. Install `spaCy-transformers`:

   ```bash
   pip install spacy-transformers
   ```

3. Download the transformer-based spaCy model:

   ```bash
   python -m spacy download en_core_web_trf
   ```

4. Install the ASC analyzer package:

   ```bash
   pip install asc-analyzer
   ```

## Quickstart
Prepare a directory with `.txt` files (e.g., `data/text/`). Each file should contain plain English text.

Then run:

```bash
asc-analyzer \
  --input-dir data/text \
  --source cow \
  --print-asc \
  --save-asc-output
````

This command will:

* Assign ASC tags to each sentence
* Print the ASC-tagged results directly to the terminal (`--print-asc`)
* Save token-level ASC tagging results as `*_ASCinfo.txt` files (`--save-asc-output`)
* Compute ASC usage statistics (e.g., diversity, proportion, frequency, and verb–ASC association strength) and save them in a CSV summary file
* The `--source` option determines which reference corpus is used for computing frequency and association measures:
    * `cow`: uses the *COW* corpus (web-based, written English)
    * `subt`: uses the *SUBTLEX* corpus (subtitle-based, spoken English)
    * Choose the source based on the register that best matches your input data.

## Options

| Option                        | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `--input-dir`          | Directory containing `.txt` files to process (default: `asc_analyzer/data/test`) |
| `--output-csv`         | Path to save the resulting CSV (default: `Written_COW.csv` or `Spoken_SubT.csv`) |
| `--source`             | Reference dataset: `cow` (written, **default**) or `subt` (spoken)          |
| `--indices`            | Comma-separated list of index names to include in the CSV (default: all standard indices) |
| `--save-asc-output`    | Save ASC-tagged outputs as `*_ASCinfo.txt` in the input directory           |
| `--print-asc`          | Print ASC-tagged results to the terminal                                    |


## Output for `--print-asc`

When using the `--print-asc` option, the output for each sentence shows aligned token information and its ASC label (`None` if no ASC applies):

```
# sent_id = 1
1	The	the	
2	idea	idea	
3	is	be	ATTR
4	trust	trust	
```
You can save this output to txt files by including `--save-asc-output`.

## Citation

- If you use the ASC tagger (`--print-asc`, `--save-asc-output`) in your research, please cite:
    - Sung, H., & Kyle, K. (2024). [Leveraging Pre-trained Language Models for Linguistic Analysis: A Case of Argument Structure Constructions](https://aclanthology.org/2024.emnlp-main.415/). *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

- The ASC Analyzer is currently in beta testing and will be updated.

# License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

See the full license [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
